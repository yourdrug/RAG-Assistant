"""Evidence-based benchmark regressions independent of external models and indexes."""

import asyncio
import csv
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain.schema import Document
from pydantic import ValidationError


from domain.services.benchmark_evaluation import evaluate_evidence, summarize_evidence
from domain.value_objects.benchmark_annotations import validate_annotations
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.rag_result import RagResult
from domain.value_objects.roles import UserKind
from infrastructure.benchmark.answer_generators import BenchmarkAnswer, RagBenchmarkGenerator
from infrastructure.benchmark.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.ml.rag.benchmark_evidence import (
    BenchmarkEvidence,
    active_evidence,
    capture_prompt,
    capture_retrieval,
)


@pytest.fixture(autouse=True)
def individual_judge_mode(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "benchmark_judge_grouped_enabled", False)


def doc(content, source="rules.pdf", **metadata):
    return {"content": content, "metadata": {"source": source, **metadata}}


FRAGMENT = {"source": "rules.pdf", "text": "Только для организаций"}


def test_correct_file_wrong_fragment_is_a_miss():
    metrics, diagnostics = evaluate_evidence(
        {"expected_fragments": [FRAGMENT]}, [doc("Пункт о другом")], [], "", "Ответ"
    )
    assert metrics["fragment_recall_at_k"] == 0
    assert metrics["fragment_mrr"] == 0
    assert diagnostics["missing_fragments"] == [FRAGMENT]


def test_exact_source_pages_and_section_are_required():
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


def test_duplicate_chunks_cannot_inflate_ndcg():
    fragments = [FRAGMENT, {"source": "rules.pdf", "text": "Срок пять дней"}]
    candidates = [doc(FRAGMENT["text"])] * 4
    metrics, _ = evaluate_evidence({"expected_fragments": fragments}, candidates, [], "", "")
    assert metrics["fragment_recall_at_k"] == 0.5
    assert 0 < metrics["fragment_ndcg_at_k"] < 1


def test_prompt_loss_is_distinct_from_retrieval_success():
    annotations = {"expected_fragments": [FRAGMENT], "required_conditions": [FRAGMENT["text"]]}
    metrics, diagnostics = evaluate_evidence(annotations, [doc(FRAGMENT["text"])], [], "Другой пункт", "")
    assert metrics["fragment_recall_at_k"] == 1
    assert metrics["context_fragment_recall"] == 0
    assert metrics["context_condition_coverage"] == 0
    assert diagnostics["missing_context_fragments"] == [FRAGMENT]


def test_parent_scope_counts_in_context_but_answer_can_drop_condition():
    annotations = {"required_facts": [["пять дней", "5 дней"]], "required_conditions": [FRAGMENT["text"]]}
    metrics, diagnostics = evaluate_evidence(
        annotations, [], [], "Только для организаций. Срок 5 дней", "Срок пять дней"
    )
    assert metrics["context_fact_coverage"] == 1
    assert metrics["answer_fact_coverage"] == 1
    assert metrics["context_condition_coverage"] == 1
    assert metrics["answer_condition_coverage"] == 0
    assert diagnostics["missing_answer_conditions"] == [FRAGMENT["text"]]


def test_unavailable_evidence_and_unannotated_cases_are_not_zero_scores():
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
def test_annotation_validation_shared_by_runner_and_api(annotations):
    from infrastructure.benchmark.runner import validate_questions
    from presentation.api.schemas.benchmark import BenchmarkQuestionCreate

    with pytest.raises(ValueError):
        validate_questions([{"question": "Q", "annotations": annotations}])
    with pytest.raises(ValidationError):
        BenchmarkQuestionCreate(question="Q", annotations=annotations)


def test_lab_annotation_mappers_and_repository_mapping():
    from domain.entities.benchmark_question import BenchmarkQuestion
    from infrastructure.database.models import BenchmarkQuestionModel
    from infrastructure.repositories.benchmark.sqlalchemy_benchmark_question_repository import (
        SQLAlchemyBenchmarkQuestionRepository,
    )
    from presentation.api.helpers import question_create_to_dto, question_to_response
    from presentation.api.schemas.benchmark import BenchmarkQuestionCreate

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
async def test_evidence_capture_is_scoped_and_concurrent():
    async def run(name):
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
async def test_full_generator_uses_actual_prompt_not_source_listing():
    class Rag:
        async def invoke(self, **kwargs):
            docs = [(Document(page_content="Полный текст", metadata={"source": "rules.pdf"}), 1)]
            capture_prompt(docs, docs, "Фактический контекст с родительскими условиями")
            return RagResult(
                answer="Ответ", sources=[{"source": "rules.pdf", "content": "Сокращённый текст"}]
            )

    result = await RagBenchmarkGenerator(Rag(), 4, None).generate(
        {"question": "Q"}, ChatContext(user_id=1, user_kind=UserKind.INTERNAL)
    )
    assert result.context == "Фактический контекст с родительскими условиями"
    assert result.evidence.retrieved[0]["content"] == "Полный текст"
    assert active_evidence.get() is None


@pytest.mark.asyncio
async def test_capture_resets_on_generation_failure():
    class Rag:
        async def invoke(self, **kwargs):
            raise RuntimeError("failed")

    with pytest.raises(RuntimeError):
        await RagBenchmarkGenerator(Rag(), 4, None).generate(
            {"question": "Q"}, ChatContext(user_id=1, user_kind=UserKind.INTERNAL)
        )
    assert active_evidence.get() is None


@pytest.mark.asyncio
async def test_case_output_contains_diagnostics_and_real_evidence(monkeypatch):
    from infrastructure.benchmark import judge, evidence_judge, metrics

    monkeypatch.setattr(
        judge,
        "judge_answer_async",
        AsyncMock(return_value={"faithfulness": 8, "relevancy": 8, "correctness": 8}),
    )
    monkeypatch.setattr(metrics, "compute_context_precision_recall", lambda *a, **k: {})
    monkeypatch.setattr(
        evidence_judge, "judge_evidence", lambda *a, **kw: {"scores": {"refusal_score": 10}, "details": {}}
    )
    from infrastructure.ml.rag.context_selection import ExclusionReason

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
    answer = BenchmarkAnswer("Ответ", evidence.context, {}, 10, 2, evidence=evidence)
    generator = SimpleNamespace(generate=AsyncMock(return_value=answer))
    result = await BenchmarkCaseEvaluator(generator, "judge").evaluate(
        1,
        {"question": "Q", "annotations": {"expected_fragments": [FRAGMENT]}},
        1,
        ChatContext(user_id=1, user_kind=UserKind.INTERNAL),
    )
    assert result["evidence_metrics"]["fragment_recall_at_k"] == 1
    assert result["evidence_metrics"]["context_fragment_recall"] == 0
    assert result["evidence"]["context"] == "Другой текст"
    assert result["evidence"]["exclusions"] == exclusions
    assert result["evidence"]["prompt_candidates"] == candidates
    assert result["evidence"]["prompt_budget"] == budget
    assert result["evidence_diagnostics"]["missing_context_fragments"] == [FRAGMENT]


def test_judge_failures_are_unavailable_with_diagnostics(monkeypatch):
    from infrastructure.benchmark import evidence_judge

    monkeypatch.setattr(evidence_judge, "get_judge_client", lambda *a: object())

    def fail(*args):
        raise RuntimeError("judge offline")

    monkeypatch.setattr(evidence_judge, "judge_with_structured_output", fail)
    result = evidence_judge.judge_evidence("Q", "A", "C", {"expected_refusal": True}, "model")
    assert result["scores"]["refusal_score"] is None
    assert result["details"]["refusal_score"]["error"] == "judge offline"


def test_legacy_annotations_skip_extra_judge(monkeypatch):
    from infrastructure.benchmark import evidence_judge

    def fail(*args):
        raise AssertionError("legacy case must not call judge")

    monkeypatch.setattr(evidence_judge, "get_judge_client", fail)
    assert evidence_judge.judge_evidence("Q", "A", "C", None, "model") == {"scores": {}, "details": {}}


def test_annotation_copy_does_not_mutate_dataset():
    annotations = {"expected_fragments": [FRAGMENT.copy()]}
    validated = validate_annotations(annotations)
    validated["expected_fragments"][0]["text"] = "new"
    assert annotations["expected_fragments"][0]["text"] == FRAGMENT["text"]


def test_csv_and_history_keep_new_metrics_and_escape_questions(tmp_path, monkeypatch):
    from infrastructure.benchmark.persistence import save_results
    from infrastructure.benchmark.benchmark_history import load_history
    from application.services.benchmark_orchestrator import compute_summary_from_results

    monkeypatch.setattr("infrastructure.benchmark.persistence.settings.data_dir", str(tmp_path))
    result = {
        "id": "q1",
        "question": 'Вопрос, с "кавычками"\nи переносом',
        "answer": "A",
        "generator_metrics": {"faithfulness": 8, "relevancy": 8, "correctness": None},
        "retriever_metrics": {"hit_rate": None, "mrr": None, "avg_similarity": 0},
        "latency_sec": 0.5,
        "evidence_metrics": {"fragment_recall_at_k": 0.5},
        "evidence": {"context_chars": 12},
    }
    save_results([result], str(tmp_path / "results"))
    with next((tmp_path / "results").glob("*.csv")).open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["question"] == result["question"]
    assert row["fragment_recall_at_k"] == "0.5"
    assert row["refusal_score"] == ""
    assert load_history(str(tmp_path))[0]["metrics"]["avg_fragment_recall_at_k"] == 0.5
    assert compute_summary_from_results([result])["avg_fragment_recall_at_k"] == 0.5


def test_real_prompt_capture_keeps_parent_scope_and_raw_retrieval():
    from domain.value_objects.llm_provider import Breadth
    from domain.value_objects.roles import UserRole
    from domain.value_objects.user_context import UserContext
    from infrastructure.ml.rag.rag_steps import step_build_context
    from infrastructure.ml.rag_pipeline import RagPipelineState
    from test_rag_pipeline import _make_rag

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
    state = RagPipelineState(
        rag=_make_rag(),
        t_pipeline_start=0,
        question="Q",
        ctx=ChatContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER),
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
    assert "Only organizations" in evidence.context
    assert "Only organizations" not in evidence.retrieved[0]["content"]
    assert any(evidence.context in message.content for message in messages)


def test_judge_client_failure_is_reported_and_missing_context_is_not_scored(monkeypatch):
    from infrastructure.benchmark import evidence_judge

    def fail(*args):
        raise RuntimeError("client offline")

    monkeypatch.setattr(evidence_judge, "get_judge_client", fail)
    result = evidence_judge.judge_evidence("Q", "A", None, {"expected_refusal": True}, "model")
    assert "citation_support_score" not in result["scores"]
    assert result["scores"]["refusal_score"] is None
    assert result["details"]["refusal_score"]["error"] == "client offline"
