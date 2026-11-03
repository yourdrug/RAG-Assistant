"""Public benchmark contracts captured before extending evaluation modes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from application.benchmark import compute_summary_from_results
from config import settings
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.rag.context import RagContext
from infrastructure.benchmark.evaluation.answer_generators import BenchmarkAnswer
from infrastructure.benchmark.evaluation.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.judging import context_judge, evidence_judge, judge
from infrastructure.ml.rag.benchmark_evidence import BenchmarkEvidence


@pytest.fixture(autouse=True)
def individual_judge_mode(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setattr(settings, "benchmark_judge_grouped_enabled", False)


@pytest.mark.asyncio
async def test_case_evaluator_preserves_question_and_actual_context(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setattr(
        judge,
        "judge_answer_async",
        AsyncMock(return_value={"faithfulness": 8, "relevancy": 7, "correctness": 9}),
    )
    monkeypatch.setattr(context_judge, "compute_context_precision_recall", lambda *_a, **_k: {})
    monkeypatch.setattr(evidence_judge, "judge_evidence", lambda *_a, **_kw: {"scores": {}, "details": {}})
    documents = [{"content": "article 17", "metadata": {"source": "law.pdf"}}]
    evidence = BenchmarkEvidence(documents, documents, "actual prompt")
    answer = BenchmarkAnswer(
        "answer",
        "actual prompt",
        {"hit_rate": 1, "mrr": 1, "avg_similarity": 0.8},
        12,
        3,
        evidence=evidence,
    )
    question = {
        "id": 17,
        "question": "question",
        "expected_answer": "expected",
        "source_hint": "law.pdf",
        "annotations": {"expected_fragments": [{"source": "law.pdf", "text": "article 17"}]},
    }
    result = await BenchmarkCaseEvaluator(
        SimpleNamespace(generate=AsyncMock(return_value=answer)), "judge"
    ).evaluate(
        1,
        question,
        2,
        RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)),
    )
    assert result["id"] == 17
    assert result["annotations"] == question["annotations"]
    assert result["evidence"]["retrieved"] == documents
    assert result["evidence"]["context"] == "actual prompt"
    assert result["run"] == 2
    assert result["generator_metrics"]["correctness"] == 9


def test_summary_preserves_fragment_coverage_and_legacy_results() -> None:
    result = {
        "id": 1,
        "question": "q",
        "answer": "a",
        "latency_sec": 1,
        "retriever_metrics": {"hit_rate": 1, "mrr": 0.5, "avg_similarity": 0.8},
        "generator_metrics": {"faithfulness": 8, "relevancy": 7, "correctness": None},
        "evidence_metrics": {"fragment_recall_at_k": 0.5, "context_fact_coverage": 0},
    }
    summary = compute_summary_from_results([result])
    assert summary["hit_rate"] == 1
    assert summary["avg_fragment_recall_at_k"] == 0.5
    assert summary["avg_context_fact_coverage"] == 0
    assert summary["fragment_recall_at_k_evaluated_count"] == 1
    assert summary["results"][0]["id"] == 1
