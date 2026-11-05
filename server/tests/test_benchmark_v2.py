"""V2 import, semantic rubric dispatch and independent retrieval regression tests."""

from __future__ import annotations

from contextlib import asynccontextmanager
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.benchmark.dataset import load_benchmark_questions
from application.benchmark.execution import BenchmarkExecutionScope, execute_benchmark
from application.benchmark.services import BenchmarkQuestionService
from application.benchmark.summary import summarize_benchmark_results
from domain.benchmark.evaluation import evaluate_retrieval
from domain.benchmark.question import ABSTENTION_TYPES, ANSWER_BEHAVIORS, answer_type, validate_annotations
from domain.shared.exceptions import ValidationError
from infrastructure.benchmark.evaluation.answer_generators import BenchmarkAnswer
from infrastructure.benchmark.evaluation.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.judging import grouped_judge
from infrastructure.ml.rag.benchmark_evidence import BenchmarkEvidence
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_question_repository import (
    SQLAlchemyBenchmarkQuestionRepository,
)
from presentation.api.helpers import question_create_to_dto, question_to_response
from presentation.api.routes.benchmark_admin import export_questions
from presentation.api.schemas.benchmark import BenchmarkQuestionsImportRequest


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from domain.benchmark.question import BenchmarkQuestion
    from domain.rag.context import RagContext
    from presentation.api.schemas.auth import CurrentUser
    from tests.fakes import FakeUnitOfWorkFactory


DATASET = Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark-questions-v2.json"


@pytest.mark.asyncio
async def test_full_v2_import_roundtrip_and_active_loading(
    fake_uow_factory: FakeUnitOfWorkFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = json.loads(DATASET.read_text())
    body = BenchmarkQuestionsImportRequest.model_validate({"questions": rows})
    stored = []

    async def store(entities: list[BenchmarkQuestion]) -> int:
        for index, entity in enumerate(entities, 1):
            entity.id = index

        stored.extend(entities)
        return len(entities)

    repository = fake_uow_factory._uow.benchmark_questions
    monkeypatch.setattr(repository, "bulk_create", AsyncMock(side_effect=store))
    monkeypatch.setattr(repository, "list_items", AsyncMock(side_effect=lambda **_kwargs: list(stored)))
    monkeypatch.setattr(
        repository,
        "list_active",
        AsyncMock(side_effect=lambda dataset: [q for q in stored if q.is_active and q.dataset == dataset]),
    )
    service = BenchmarkQuestionService(fake_uow_factory)
    assert await service.bulk_create([question_create_to_dto(q) for q in body.questions], 1) == 202
    exported = await export_questions(_admin=cast("CurrentUser", SimpleNamespace(id=1)), service=service)
    assert len(exported) == 202
    by_key = {(q["case_id"], q["dataset"]): q for q in exported}

    for original in rows:
        exported_case = by_key[original["case_id"], original["dataset"]]

        for key, value in original.items():
            assert exported_case[key] == value

    active = await load_benchmark_questions(fake_uow_factory, "main")
    assert len(active) == 171
    assert "rag-0023" not in {q["case_id"] for q in active}
    quick = await load_benchmark_questions(fake_uow_factory, "quick-30")
    assert len(quick) == 30

    for entity in await service.export():
        mapped = question_to_response(entity)
        assert mapped.case_id == entity.case_id
        assert mapped.question_style == entity.question_style


@pytest.mark.asyncio
async def test_v2_identity_reaches_sql_storage(fake_uow_factory: FakeUnitOfWorkFactory) -> None:
    rows = json.loads(DATASET.read_text())
    body = BenchmarkQuestionsImportRequest.model_validate({"questions": rows})
    session = MagicMock(flush=AsyncMock())
    fake_uow_factory._uow.benchmark_questions = SQLAlchemyBenchmarkQuestionRepository(session)
    await BenchmarkQuestionService(fake_uow_factory).bulk_create(
        [question_create_to_dto(q) for q in body.questions], 1
    )

    for orm, row in zip(session.add_all.call_args.args[0], rows, strict=True):
        entity = SQLAlchemyBenchmarkQuestionRepository._to_entity(orm)
        assert entity.case_id == row["case_id"]
        assert entity.annotations == row["annotations"]
        assert entity.variant_of == row.get("variant_of")
        assert entity.question_style == row["question_style"]


@pytest.mark.parametrize("kind", ANSWER_BEHAVIORS)
def test_semantic_dispatch_for_all_six_types(kind: Any) -> None:
    annotations = {"answer_type": kind, "expected_behavior": ANSWER_BEHAVIORS[kind]}
    groups = grouped_judge.judge_groups(
        {"question": "Q", "expected_answer": "reference", "annotations": annotations}, "A", "C", "C"
    )
    payload, metrics = groups[0]
    assert payload["answer_type"] == kind
    assert payload["expected_behavior"] == ANSWER_BEHAVIORS[kind]
    assert payload["answer_rubric"]
    assert "correctness" in metrics
    assert "refusal_score" in metrics
    assert (len(groups) == 2) is (kind in ABSTENTION_TYPES)


def test_legacy_fallback() -> None:
    assert answer_type(None) == "factual"
    assert answer_type({"expected_refusal": True}) == "not_in_corpus"


@pytest.mark.parametrize(
    "annotations",
    [
        {"answer_type": "invented"},
        {"answer_type": "negative", "expected_refusal": True},
        {"answer_type": "false_premise", "expected_behavior": "abstain"},
        {"source_files": "rules.pdf"},
        {"gold_evidence": [{"source": "rules.pdf"}]},
        {"review_required": "false"},
    ],
)
def test_invalid_v2_annotations_rejected(annotations: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        validate_annotations(annotations)


@pytest.mark.parametrize("kind", ["factual", "negative", "false_premise", "ambiguous"])
def test_gold_evidence_has_priority_and_filename_alone_cannot_pass(kind: str) -> None:
    question = {
        "source_hint": "rules.pdf",
        "annotations": {
            "answer_type": kind,
            "gold_evidence": [{"source": "rules.pdf", "text": "complete evidence", "anchor": "complete"}],
            "expected_fragments": [{"source": "rules.pdf", "text": "old quote"}],
        },
    }

    def score(text: str) -> dict[str, Any]:
        return evaluate_retrieval(question, [{"content": text, "metadata": {"source": "rules.pdf"}}])

    assert score("old quote")["hit_rate"] == 0
    assert score("old quote")["source_hit_rate"] == 1
    assert score("complete evidence")["hit_rate"] == 1
    assert score("complete evidence")["fragment_recall_at_k"] == 1


@pytest.mark.parametrize("kind", ABSTENTION_TYPES)
def test_abstention_has_no_retrieval_denominator(kind: Any) -> None:
    result = evaluate_retrieval(
        {
            "source_hint": "rules.pdf",
            "annotations": {
                "answer_type": kind,
                "gold_evidence": [{"source": "rules.pdf", "text": "quote"}],
            },
        },
        [],
    )
    assert result["hit_rate"] is None
    assert result["mrr"] is None
    assert result["labelled"] is False


@pytest.mark.asyncio
async def test_v2_report_and_no_reference_leak_to_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    async def judge(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            key: {"score": 10, "reason": "ok"}
            for key in ("faithfulness", "relevancy", "correctness", "refusal_score", "citation_support_score")
        }

    monkeypatch.setattr(grouped_judge, "judge_case_grouped", judge)
    generator = SimpleNamespace(
        generate=AsyncMock(
            return_value=BenchmarkAnswer(
                "corrected premise", "quote", {}, None, None, evidence=BenchmarkEvidence([], [], "quote")
            )
        )
    )
    case = {
        "question": "Q",
        "case_id": "rag-test",
        "variant_of": "rag-original",
        "question_style": "natural",
        "dataset": "main",
        "expected_answer": "secret gold",
        "annotations": {"answer_type": "false_premise"},
    }
    result = await BenchmarkCaseEvaluator(generator, "judge").evaluate(
        1, case, 1, cast("RagContext", SimpleNamespace())
    )
    assert generator.generate.call_args.args[0] == {"question": "Q"}
    assert result["case_id"] == "rag-test"
    assert result["answer_type"] == "false_premise"
    assert result["variant_of"] == "rag-original"
    assert result["question_style"] == "natural"
    assert result["generator_metrics"]["correctness"] == 10


def test_summary_deduplicates_dataset_copies_preserves_runs_and_groups() -> None:
    row = {
        "case_id": "rag-1",
        "dataset": "main",
        "question_style": "natural",
        "run": 1,
        "annotations": {"answer_type": "negative"},
        "expected_answer": "No",
        "generator_metrics": {"faithfulness": 10, "relevancy": 10, "correctness": 10},
        "retriever_metrics": {"hit_rate": 1, "mrr": 1, "avg_similarity": 1},
        "latency_sec": 1,
    }
    quick: dict[str, Any] = {**copy.deepcopy(row), "dataset": "quick-30"}
    quick["generator_metrics"]["correctness"] = 0
    summary = summarize_benchmark_results([quick, row, {**row, "run": 2}])
    assert summary["total_questions"] == 2
    assert summary["dataset_duplicate_count"] == 1
    assert summary["avg_correctness"] == 10
    assert summary["by_answer_type"]["negative"]["total_questions"] == 2
    assert summary["by_question_style"]["natural"]["total_questions"] == 2
    assert summary["by_dataset"]["main"]["total_questions"] == 2
    assert summary["by_dataset"]["quick-30"]["avg_correctness"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("only_inactive", [True, False])
async def test_execution_never_runs_inactive_cases(only_inactive: bool) -> None:
    evaluator = SimpleNamespace(evaluate=AsyncMock(return_value={"id": 2}))

    @asynccontextmanager
    async def scope() -> AsyncIterator[BenchmarkExecutionScope]:
        yield BenchmarkExecutionScope(evaluator=evaluator, corpus_metadata={})

    questions = [{"question": "Disabled", "is_active": False}]

    if not only_inactive:
        questions.append({"question": "Enabled", "id": 2})

    kwargs = {
        "questions": questions,
        "top_k": 2,
        "judge_model": "judge",
        "scope": scope,
        "reporter": MagicMock(),
        "settings_identity": "test",
        "rubric_version": "v2",
    }

    if only_inactive:
        with pytest.raises(ValueError, match="at least one active question"):
            await execute_benchmark(**cast("dict[str, Any]", kwargs))
        evaluator.evaluate.assert_not_awaited()
    else:
        assert await execute_benchmark(**cast("dict[str, Any]", kwargs)) == [{"id": 2}]
        evaluator.evaluate.assert_awaited_once()
        assert evaluator.evaluate.call_args.args[1]["question"] == "Enabled"
