"""Observed stage losses, retry isolation, ACL safety and instrumentation parity."""

import asyncio
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from langchain_core.documents import Document
import pytest

from domain.benchmark.evaluation import evaluate_stage_evidence
from domain.benchmark.question import validate_annotations
from domain.benchmark.stage_reporting import compare_stage_runs, summarize_stage_tags
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.rag.context import RagContext
from domain.rag.result import RagResult
from domain.rag.settings import RetrieverConfig
from infrastructure.benchmark.evaluation import case_evaluator
from infrastructure.benchmark.evaluation.annotation_validation import AnnotationValidator
from infrastructure.benchmark.evaluation.answer_generators import BenchmarkAnswer, RagBenchmarkGenerator
from infrastructure.ml.rag import rag_steps, retrieval_pipeline
from infrastructure.ml.rag.benchmark_evidence import (
    BenchmarkEvidence,
    active_evidence,
    authorize_capture,
    begin_attempt,
    capture_exclusions,
    capture_prompt,
    capture_stage,
    snapshot_documents,
)
from infrastructure.ml.rag.context_selection import ExclusionReason, exclusion_record
from infrastructure.ml.rag.rag_reranking import rerank_documents
from rag_stage_fixtures import build_context, prepared_query, stage_fixture  # type: ignore[import-not-found]
from test_rag_pipeline import _make_rag  # type: ignore[import-not-found]


CTX = RagContext(UserContext(1, UserKind.INTERNAL, UserRole.USER))
FRAGMENTS = [
    {"source": "rules.pdf", "text": text} for text in ("Fact A", "Only if B", "Exception C", "Rule D")
]
ANNOTATIONS = {"expected_fragments": FRAGMENTS, "required_conditions": ["Only if B", "Exception C"]}


def documents() -> list[tuple[Document, float]]:
    return [
        (
            Document(
                page_content=f["text"], metadata={"source": "rules.pdf", "document_id": 1, "chunk_index": i}
            ),
            1 - i / 10,
        )
        for i, f in enumerate(FRAGMENTS)
    ]


def test_observed_losses_recovery_and_lexical_omission() -> None:
    docs = documents()
    evidence = BenchmarkEvidence()
    token = active_evidence.set(evidence)
    try:
        begin_attempt("Q", "search Q")
        authorize_capture(docs)
        capture_stage("candidates", docs, ordered=False)
        capture_stage("reranked", docs)
        capture_exclusions([exclusion_record(docs[3], ExclusionReason.FINAL_TOP_K, "retrieval")])
        capture_stage("retained", docs[:3])
        capture_stage("enriched", docs, ordered=False)
        capture_prompt(
            docs,
            docs[:3],
            "Fact A Only if B Exception C",
            exclusions=[exclusion_record(docs[3], ExclusionReason.BUDGET, "context")],
        )
    finally:
        active_evidence.reset(token)
    result = evaluate_stage_evidence(ANNOTATIONS, asdict(evidence), "Fact A Only if B")
    stages = result["attempts"][0]["stages"]
    assert [
        stages[name]["coverage"] for name in ("candidates", "reranked", "retained", "enriched", "prompt")
    ] == [1, 1, 0.75, 1, 0.75]
    assert stages["candidates"]["mrr"] is None
    assert stages["reranked"]["mrr"] == 1
    assert stages["retained"]["lost_evidence"] == [3]
    assert stages["enriched"]["recovered_evidence"] == [3]
    losses = [issue for issue in result["issues"] if issue["kind"] == "evidence_lost"]
    assert losses[-1]["decisions"][-1]["reason"] == "budget"
    assert any(
        issue["kind"] == "lexical_answer_omission" and issue["requirement"] == "Exception C"
        for issue in result["issues"]
    )
    assert result["first_observed_loss"]["3"]["to"] == "retained"


@pytest.mark.asyncio
async def test_real_reranker_records_threshold_and_top_n_without_changing_results() -> None:
    docs = [doc for doc, _ in documents()]
    reranker = SimpleNamespace(predict=lambda _pairs: [0.9, 0.8, 0.7, 0.01])
    control = await rerank_documents("Q", docs, 2, reranker, min_score=0.1)
    evidence = BenchmarkEvidence()
    token = active_evidence.set(evidence)
    try:
        begin_attempt("Q", "Q")
        authorize_capture(docs)
        capture_stage("candidates", docs, ordered=False)
        observed = await rerank_documents("Q", docs, 2, reranker, min_score=0.1)
    finally:
        active_evidence.reset(token)
    assert observed == control
    attempt = evidence.attempts[0]
    assert len(attempt["stages"]["reranked"]["documents"]) == 3
    assert {item["reason"] for item in attempt["exclusions"]} == {"reranker_threshold", "reranker_top_n"}


def test_retry_preserves_attempts_and_missing_prompt_is_unexecuted() -> None:
    evidence = BenchmarkEvidence()
    token = active_evidence.set(evidence)
    docs = documents()
    try:
        begin_attempt("Q", "first query")
        capture_stage("candidates", docs[:1], ordered=False)
        begin_attempt("Q", "refined query")
        capture_stage("candidates", docs, ordered=False)
    finally:
        active_evidence.reset(token)
    result = evaluate_stage_evidence(ANNOTATIONS, asdict(evidence), "")
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["stages"]["candidates"]["coverage"] == 0.25
    assert result["attempts"][1]["stages"]["candidates"]["recovered_evidence"] == [1, 2, 3]
    assert result["attempts"][1]["stages"]["prompt"] == {"status": "not_executed", "coverage": None}
    assert evidence.attempts[0]["query"] == "first query"


def test_empty_missing_legacy_and_invalid_are_distinct() -> None:
    assert evaluate_stage_evidence(ANNOTATIONS, {}, "")["attempts"] == []
    assert BenchmarkEvidence(retrieved=[], selected=[], context="").schema_version == 1
    evidence: dict[str, Any] = {
        "attempts": [{"stages": {"candidates": {"status": "executed", "documents": [], "ordered": False}}}]
    }
    assert (
        evaluate_stage_evidence(ANNOTATIONS, evidence, "")["attempts"][0]["stages"]["candidates"]["coverage"]
        == 0
    )
    assert (
        evaluate_stage_evidence(None, evidence, "")["attempts"][0]["stages"]["candidates"]["coverage"] is None
    )
    evidence["annotation_validation"] = {"status": "invalid_annotation"}
    assert (
        evaluate_stage_evidence(ANNOTATIONS, evidence, "")["attempts"][0]["stages"]["candidates"]["coverage"]
        is None
    )


def test_equivalent_sources_count_once_and_version_is_enforced() -> None:
    annotations = validate_annotations(
        {
            "expected_fragments": [
                {
                    "alternatives": [
                        FRAGMENTS[0],
                        {"source": "other.pdf", "text": "Equivalent fact", "act_version_id": 3},
                    ]
                }
            ]
        }
    )
    evidence: dict[str, Any] = {
        "attempts": [
            {
                "stages": {
                    "candidates": {
                        "status": "executed",
                        "ordered": False,
                        "documents": [
                            {
                                "content": "Equivalent fact",
                                "metadata": {"source": "other.pdf", "act_version_id": 3},
                            }
                        ],
                    }
                }
            }
        ]
    }
    assert (
        evaluate_stage_evidence(annotations, evidence, "")["attempts"][0]["stages"]["candidates"]["coverage"]
        == 1
    )
    evidence["attempts"][0]["stages"]["candidates"]["documents"][0]["metadata"]["act_version_id"] = 4
    assert (
        evaluate_stage_evidence(annotations, evidence, "")["attempts"][0]["stages"]["candidates"]["coverage"]
        == 0
    )


def test_snapshots_are_immutable_and_unauthorized_exclusions_are_not_saved() -> None:
    docs = documents()
    evidence = BenchmarkEvidence()
    token = active_evidence.set(evidence)
    try:
        begin_attempt("Q", "Q")
        authorize_capture(docs[:1])
        capture_stage("candidates", docs[:1], ordered=False)
        capture_exclusions([exclusion_record(docs[1], ExclusionReason.DUPLICATE, "retrieval")])
        docs[0][0].page_content = "mutated"
    finally:
        active_evidence.reset(token)
    assert evidence.attempts[0]["stages"]["candidates"]["documents"][0]["content"] == "Fact A"
    assert evidence.exclusions == []
    assert snapshot_documents([docs[0][0]])[0]["score"] is None


@pytest.mark.asyncio
async def test_parallel_generation_isolated_and_history_date_forwarded() -> None:
    async def invoke(**kwargs: Any) -> RagResult:
        assert set(kwargs) == {"question", "history", "ctx"}
        assert kwargs["history"] == [{"role": "user", "content": "previous"}]
        assert kwargs["ctx"].as_of_date.isoformat() == "2020-01-01"
        begin_attempt(kwargs["question"], kwargs["question"])
        await asyncio.sleep(0)
        return RagResult(answer=kwargs["question"], sources=[])

    generator = RagBenchmarkGenerator(SimpleNamespace(invoke=invoke), 4, None)
    results = await asyncio.gather(
        *(
            generator.generate(
                {
                    "question": str(i),
                    "history": [{"role": "user", "content": "previous"}],
                    "as_of_date": "2020-01-01",
                    "annotations": ANNOTATIONS,
                },
                CTX,
            )
            for i in range(3)
        )
    )
    assert [
        evidence.attempts[0]["question"] for result in results if (evidence := result.evidence) is not None
    ] == ["0", "1", "2"]
    assert active_evidence.get() is None


@pytest.mark.asyncio
async def test_missing_reference_is_invalid_annotation() -> None:
    search = SimpleNamespace(search_substring=AsyncMock(return_value=[]))
    result = await AnnotationValidator(search).validate(ANNOTATIONS, CTX)
    assert result["status"] == "invalid_annotation"
    assert result["missing_fragments"] == [0, 1, 2, 3]
    assert all(call.args[1] == CTX.user for call in search.search_substring.call_args_list)


def test_comparison_excludes_changed_annotations_and_averages_repetitions() -> None:
    cases = [
        {
            "id": i,
            "question": f"Q{i}",
            "annotations": ANNOTATIONS,
            "tags": ["conditions"],
            "evidence_metrics": {"answer_condition_coverage": 0.5},
        }
        for i in range(3)
    ]
    candidates = [{**case, "evidence_metrics": {"answer_condition_coverage": 1}} for case in cases]
    candidates[2] = {**candidates[2], "annotations": None}
    comparison = compare_stage_runs([*cases, cases[0]], candidates, samples=100)
    metric = comparison["metrics"]["answer_condition_coverage"]
    assert metric["paired_case_count"] == 2
    assert metric["mean_delta"] == 0.5
    assert metric["ci95"] == [0.5, 0.5]
    assert len(comparison["excluded"]) == 1
    assert summarize_stage_tags(candidates)["by_tags"]["conditions"]["case_count"] == 3


@pytest.mark.asyncio
async def test_pipeline_top_k_neighbor_recovery_acl_and_prompt_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = documents()
    denied = Document(page_content="PRIVATE document", metadata={"source": "private.pdf", "document_id": 9})
    access = SimpleNamespace(allowed_document_ids=AsyncMock(return_value={1}))
    ml = SimpleNamespace(
        reranker_semaphore=asyncio.Semaphore(1),
        reranker=lambda: SimpleNamespace(predict=lambda pairs: [0.9, 0.8, 0.7, 0.6][: len(pairs)]),
    )

    async def neighbors(
        selected: list[tuple[Document, float]], *args: Any, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        return [*selected, docs[2], (denied, 0.7)]

    monkeypatch.setattr(retrieval_pipeline, "enrich_with_neighbors", neighbors)

    async def run(capture: bool) -> tuple[Any, Any, Any]:
        state = stage_fixture(
            rag=_make_rag(retriever=RetrieverConfig(fetch_k=10, top_k=2, fetch_k_broad=10, top_k_broad=4)),
            question="Q",
            ctx=CTX,
            user=CTX.user,
            access_filter=None,
            retrieval_filter=None,
            query_domain="general",
            rerank_top_n=4,
        )
        evidence = BenchmarkEvidence() if capture else None
        token = active_evidence.set(evidence)
        try:
            begin_attempt("Q", "Q")
            state.docs, _, _ = await retrieval_pipeline.rerank_and_enrich(
                "Q", [*(d for d, _ in docs), denied], state, ml, None, document_access=access
            )
            messages, _ = build_context(state, None)
            return state.docs, messages, evidence
        finally:
            active_evidence.reset(token)

    control, control_messages, _ = await run(False)
    observed, messages, evidence = await run(True)
    assert (observed, messages) == (control, control_messages)
    stages = evidence.attempts[0]["stages"]
    assert [len(stages[name]["documents"]) for name in ("candidates", "retained", "enriched")] == [4, 2, 3]
    assert "PRIVATE document" not in str(asdict(evidence))
    assert stages["prompt"]["status"] == "prepared"
    assert any(evidence.context in message.content for message in messages)
    assert any(item["reason"] == "final_top_k" for item in evidence.exclusions)


@pytest.mark.asyncio
async def test_diagnostic_mode_bypasses_cache_read_and_write(monkeypatch: pytest.MonkeyPatch) -> None:
    rag = _make_rag()
    state = stage_fixture(
        question="Q", ctx=CTX, rag=replace(rag, features=replace(rag.features, cache_enabled=True))
    )
    query = prepared_query(state)
    monkeypatch.setattr(rag_steps, "get_corpus_revision", AsyncMock(side_effect=AssertionError("cache read")))
    monkeypatch.setattr(rag_steps, "store_answer_cache", AsyncMock(side_effect=AssertionError("cache write")))
    operations = rag_steps.RagOperations(SimpleNamespace(), document_access=SimpleNamespace())
    token = active_evidence.set(BenchmarkEvidence(schema_version=2))
    try:
        assert await operations.lookup(query) is query
        completed: Any = SimpleNamespace(
            answer=SimpleNamespace(context=SimpleNamespace(retrieval=SimpleNamespace(prepared=query)))
        )
        await operations.store(completed)
    finally:
        active_evidence.reset(token)


@pytest.mark.asyncio
async def test_invalid_annotation_is_redacted_and_references_never_reach_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def generate(question: dict[str, Any], ctx: RagContext) -> BenchmarkAnswer:
        assert question == {"question": "Q"}
        return BenchmarkAnswer("A", "", {}, 0, 0, evidence=BenchmarkEvidence(schema_version=2))

    judge = AsyncMock(side_effect=AssertionError("invalid annotations must not be judged"))
    monkeypatch.setattr(case_evaluator, "judge_generated", judge)
    evaluator = case_evaluator.BenchmarkCaseEvaluator(
        SimpleNamespace(generate=generate),
        "judge",
        annotation_validator=AnnotationValidator(
            SimpleNamespace(search_substring=AsyncMock(return_value=[]))
        ),
    )
    result = await evaluator.evaluate(
        1,
        {
            "question": "Q",
            "expected_answer": "PRIVATE reference",
            "annotations": {"expected_fragments": [{"source": "private.pdf", "text": "PRIVATE reference"}]},
        },
        1,
        CTX,
    )
    assert result["annotation_validation"]["status"] == "invalid_annotation"
    assert result["annotations"] is None
    assert result["expected_answer"] is None
    assert "PRIVATE reference" not in str(result)
    assert result["evidence_metrics"]["fragment_recall_at_k"] is None
