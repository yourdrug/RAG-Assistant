"""Evidence-based benchmark regressions independent of external models and indexes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock

from langchain.schema import Document
from pydantic import ValidationError
import pytest

from config import settings
from domain.benchmark.evaluation import evaluate_evidence, summarize_evidence
from domain.benchmark.question import BenchmarkQuestion, validate_annotations
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.rag.breadth import Breadth
from domain.rag.context import RagContext
from domain.rag.result import RagResult
from infrastructure.benchmark.evaluation.answer_generators import BenchmarkAnswer, RagBenchmarkGenerator
from infrastructure.benchmark.evaluation.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.evaluation.runner import validate_questions
from infrastructure.benchmark.judging import context_judge, evidence_judge, judge
from infrastructure.database.models import BenchmarkQuestionModel
from infrastructure.ml.rag.benchmark_evidence import (
    BenchmarkEvidence,
    active_evidence,
    capture_prompt,
    capture_retrieval,
)
from infrastructure.ml.rag.context_selection import ExclusionReason
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_question_repository import (
    SQLAlchemyBenchmarkQuestionRepository,
)
from presentation.api.helpers import question_create_to_dto, question_to_response
from presentation.api.schemas.benchmark import BenchmarkQuestionCreate
from tests.fakes import require_present
from tests.rag_stage_fixtures import build_context as step_build_context, stage_fixture
from tests.test_rag_pipeline import make_rag


if TYPE_CHECKING:
    from application.chat.ports import ChatRAGPort


@pytest.fixture(autouse=True)
def individual_judge_mode(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setattr(settings, "benchmark_judge_grouped_enabled", False)


def doc(content: Any, source: str = "rules.pdf", **metadata: Any) -> dict[str, Any]:
    return {"content": content, "metadata": {"source": source, **metadata}}


FRAGMENT = {"source": "rules.pdf", "text": "Только для организаций"}


def test_correct_file_wrong_fragment_is_a_miss() -> None:
    metrics, diagnostics = evaluate_evidence(
        {"expected_fragments": [FRAGMENT]}, [doc("Пункт о другом")], [], "", "Ответ"
    )
    assert metrics["fragment_recall_at_k"] == 0
    assert metrics["fragment_mrr"] == 0
    assert diagnostics["missing_fragments"] == [FRAGMENT]


def test_exact_source_pages_and_section_are_required() -> None:
    fragment = {**FRAGMENT, "pages": [2], "section": "Пункт 4"}
    candidates = [
        doc(FRAGMENT["text"], "rules.pdf.bak", page=2, section="Пункт 4"),
        doc(FRAGMENT["text"], page=1, section="Пункт 4"),
        doc(FRAGMENT["text"], page=2, section="Пункт 5"),
        doc("Только\nдля ОРГАНИЗАЦИЙ", "/data/rules.pdf", page_start=2, page_end=3, section="Пункт 4"),
    ]
    metrics, _ = evaluate_evidence({"expected_fragments": [fragment]}, candidates, [], "", "")
    assert metrics["fragment_recall_at_k"] == 1
    assert metrics["fragment_mrr"] == 0.25
    assert metrics["fragment_ndcg_at_k"] == pytest.approx(1 / 2.321928094887362)


def test_duplicate_chunks_cannot_inflate_ndcg() -> None:
    fragments = [FRAGMENT, {"source": "rules.pdf", "text": "Срок пять дней"}]
    candidates = [doc(FRAGMENT["text"])] * 4
    metrics, _ = evaluate_evidence({"expected_fragments": fragments}, candidates, [], "", "")
    assert metrics["fragment_recall_at_k"] == 0.5
    assert 0 < metrics["fragment_ndcg_at_k"] < 1


def test_prompt_loss_is_distinct_from_retrieval_success() -> None:
    annotations = {"expected_fragments": [FRAGMENT], "required_conditions": [FRAGMENT["text"]]}
    metrics, diagnostics = evaluate_evidence(annotations, [doc(FRAGMENT["text"])], [], "Другой пункт", "")
    assert metrics["fragment_recall_at_k"] == 1
    assert metrics["context_fragment_recall"] == 0
    assert metrics["context_condition_coverage"] == 0
    assert diagnostics["missing_context_fragments"] == [FRAGMENT]


def test_parent_scope_counts_in_context_but_answer_can_drop_condition() -> None:
    annotations = {"required_facts": [["пять дней", "5 дней"]], "required_conditions": [FRAGMENT["text"]]}
    metrics, diagnostics = evaluate_evidence(
        annotations, [], [], "Только для организаций. Срок 5 дней", "Срок пять дней"
    )
    assert metrics["context_fact_coverage"] == 1
    assert metrics["answer_fact_coverage"] == 1
    assert metrics["context_condition_coverage"] == 1
    assert metrics["answer_condition_coverage"] == 0
    assert diagnostics["missing_answer_conditions"] == [FRAGMENT["text"]]


def test_unavailable_evidence_and_unannotated_cases_are_not_zero_scores() -> None:
    metrics, _ = evaluate_evidence({"expected_fragments": [FRAGMENT]}, None, None, None, "")
    assert metrics["fragment_recall_at_k"] is None
    assert metrics["context_fragment_recall"] is None
    metrics, _ = evaluate_evidence(None, [], [], "", "")
    assert all(value is None for value in metrics.values())
    summary = summarize_evidence(
        [{"evidence_metrics": metrics}, {"evidence_metrics": {"fragment_recall_at_k": 0}}]
    )
    assert summary["avg_fragment_recall_at_k"] == 0
    assert summary["fragment_recall_at_k_evaluated_count"] == 1


@pytest.mark.parametrize(
    "annotations",
    [
        {"expected_refusal": "true"},
        {"required_facts": [""]},
        {"expected_fragments": [{"source": "x"}]},
        {"expected_fragments": [{**FRAGMENT, "pages": [True]}]},
        {"unsupported": True},
    ],
)
def test_annotation_validation_shared_by_runner_and_api(annotations: dict[str, Any]) -> None:

    with pytest.raises(ValueError, match=r"(annotations|expected_refusal|required_facts|fragment)"):
        validate_questions([{"question": "Q", "annotations": annotations}])
    with pytest.raises(ValidationError):
        BenchmarkQuestionCreate(question="Q", annotations=annotations)


def test_lab_annotation_mappers_and_repository_mapping() -> None:

    annotations = {"expected_fragments": [FRAGMENT], "expected_refusal": False}
    dto = question_create_to_dto(BenchmarkQuestionCreate(question="Q", annotations=annotations))
    assert dto.annotations == annotations
    entity = BenchmarkQuestion(question="Q", annotations=annotations, id=1)
    assert question_to_response(entity).annotations == annotations
    orm = BenchmarkQuestionModel(
        id=1,
        question="Q",
        annotations=annotations,
        dataset=entity.dataset,
        is_active=True,
        creation_date=entity.creation_date,
    )
    assert SQLAlchemyBenchmarkQuestionRepository._to_entity(orm).annotations == annotations


@pytest.mark.asyncio
async def test_evidence_capture_is_scoped_and_concurrent() -> None:
    async def run(name: str) -> BenchmarkEvidence:
        evidence = BenchmarkEvidence()
        token = active_evidence.set(evidence)
        try:
            capture_retrieval([(Document(page_content=name, metadata={"source": name}), 1)])
            await asyncio.sleep(0)
            capture_prompt([(Document(page_content=name), 1)], [], name)
            return evidence
        finally:
            active_evidence.reset(token)

    left, right = await asyncio.gather(run("left"), run("right"))
    assert left.context == "left"
    assert right.context == "right"
    assert active_evidence.get() is None


@pytest.mark.asyncio
async def test_full_generator_uses_actual_prompt_not_source_listing() -> None:
    class Rag:
        async def invoke(self, **kwargs: Any) -> Any:
            _ = kwargs
            docs = [(Document(page_content="Полный текст", metadata={"source": "rules.pdf"}), 1)]
            capture_prompt(docs, docs, "Фактический контекст с родительскими условиями")
            return RagResult(
                answer="Ответ", sources=[{"source": "rules.pdf", "content": "Сокращённый текст"}]
            )

    result = await RagBenchmarkGenerator(cast("ChatRAGPort", Rag()), 4, None).generate(
        {"question": "Q"},
        RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)),
    )
    assert result.context == "Фактический контекст с родительскими условиями"
    assert (
        require_present(require_present(require_present(result.evidence).retrieved)[0])["content"]
        == "Полный текст"
    )
    assert active_evidence.get() is None


@pytest.mark.asyncio
async def test_capture_resets_on_generation_failure() -> None:
    class Rag:
        async def invoke(self, **kwargs: Any) -> None:
            _ = kwargs
            msg = "failed"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError):
        await RagBenchmarkGenerator(cast("ChatRAGPort", Rag()), 4, None).generate(
            {"question": "Q"},
            RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)),
        )
    assert active_evidence.get() is None


@pytest.mark.asyncio
async def test_case_output_contains_diagnostics_and_real_evidence(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setattr(
        judge,
        "judge_answer_async",
        AsyncMock(return_value={"faithfulness": 8, "relevancy": 8, "correctness": 8}),
    )
    monkeypatch.setattr(context_judge, "compute_context_precision_recall", lambda *_a, **_k: {})
    monkeypatch.setattr(
        evidence_judge, "judge_evidence", lambda *_a, **_kw: {"scores": {"refusal_score": 10}, "details": {}}
    )

    exclusions = [{"reason": ExclusionReason.BUDGET.value, "stage": "context"}]
    candidates = [doc(FRAGMENT["text"])]
    budget = {"input_tokens": 100, "num_ctx": 256}
    evidence = BenchmarkEvidence(
        candidates,
        [],
        "Другой текст",
        exclusions=exclusions,
        prompt_candidates=candidates,
        prompt_budget=budget,
    )
    answer = BenchmarkAnswer("Ответ", require_present(evidence.context), {}, 10, 2, evidence=evidence)
    generator = SimpleNamespace(generate=AsyncMock(return_value=answer))
    result = await BenchmarkCaseEvaluator(generator, "judge").evaluate(
        1,
        {"question": "Q", "annotations": {"expected_fragments": [FRAGMENT]}},
        1,
        RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)),
    )
    assert result["evidence_metrics"]["fragment_recall_at_k"] == 1
    assert result["evidence_metrics"]["context_fragment_recall"] == 0
    assert result["evidence"]["context"] == "Другой текст"
    assert result["evidence"]["exclusions"] == exclusions
    assert result["evidence"]["prompt_candidates"] == candidates
    assert result["evidence"]["prompt_budget"] == budget
    assert result["evidence_diagnostics"]["missing_context_fragments"] == [FRAGMENT]


def test_judge_failures_are_unavailable_with_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setattr(evidence_judge, "get_judge_client", lambda *_a, **_kwargs: object())

    def fail(*args: Any) -> None:
        msg = "judge offline"
        raise RuntimeError(msg)

    monkeypatch.setattr(evidence_judge, "judge_with_structured_output", fail)
    result = evidence_judge.judge_evidence("Q", "A", "C", {"expected_refusal": True}, "model")
    assert result["scores"]["refusal_score"] is None
    assert result["details"]["refusal_score"]["error"] == "judge offline"


def test_legacy_annotations_skip_extra_judge(monkeypatch: pytest.MonkeyPatch) -> None:

    def fail(*args: Any) -> None:
        msg = "legacy case must not call judge"
        raise AssertionError(msg)

    monkeypatch.setattr(evidence_judge, "get_judge_client", fail)
    assert evidence_judge.judge_evidence("Q", "A", "C", None, "model") == {"scores": {}, "details": {}}


def test_annotation_copy_does_not_mutate_dataset() -> None:
    annotations = {"expected_fragments": [FRAGMENT.copy()]}
    validated = validate_annotations(annotations)
    require_present(require_present(require_present(validated)["expected_fragments"])[0])["text"] = "new"
    assert annotations["expected_fragments"][0]["text"] == FRAGMENT["text"]


def test_real_prompt_capture_keeps_parent_scope_and_raw_retrieval() -> None:

    raw = [(Document(page_content="Submit application", metadata={"source": "rules.pdf"}), 0.9)]
    enriched = [
        (
            Document(
                page_content="Submit application",
                metadata={
                    "source": "rules.pdf",
                    "parent_units": [{"content": "Only organizations", "unit_kind": "preamble"}],
                },
            ),
            0.9,
        )
    ]
    state = stage_fixture(
        rag=make_rag(),
        t_pipeline_start=0,
        question="Q",
        ctx=RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)),
        user=UserContext(1, UserKind.INTERNAL, UserRole.USER),
        access_filter=None,
        retrieval_filter=None,
        breadth=Breadth.NARROW,
        docs=enriched,
    )
    evidence = BenchmarkEvidence()
    token = active_evidence.set(evidence)
    try:
        capture_retrieval(raw)
        messages, _ = step_build_context(state, None)
    finally:
        active_evidence.reset(token)
    assert "Only organizations" in require_present(evidence.context)
    assert "Only organizations" not in require_present(require_present(evidence.retrieved)[0])["content"]
    assert any(evidence.context in message.content for message in messages)


def test_judge_client_failure_is_reported_and_missing_context_is_not_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def fail(*args: Any, runtime: object = None) -> None:
        msg = "client offline"
        raise RuntimeError(msg)

    monkeypatch.setattr(evidence_judge, "get_judge_client", fail)
    result = evidence_judge.judge_evidence("Q", "A", None, {"expected_refusal": True}, "model")
    assert "citation_support_score" not in result["scores"]
    assert result["scores"]["refusal_score"] is None
    assert result["details"]["refusal_score"]["error"] == "client offline"
