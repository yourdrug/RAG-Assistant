"""Characterization tests for benchmark/benchmark.py — 1088-line god-file.

Locks down the pure-logic functions BEFORE refactoring.
LLM judge calls and Qdrant/retrieval are fully mocked.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from langchain.schema import Document  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(content: str, source: str = "doc.pdf", filename: str = "") -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "filename": filename},
    )


def _result(
    faithfulness=8.0,
    relevancy=7.0,
    correctness=None,
    hit_rate=1,
    mrr=1.0,
    avg_sim=0.85,
    latency=1.5,
    context_precision=8.0,
    context_recall=7.0,
) -> dict:
    return {
        "id": "q1",
        "question": "test question",
        "answer": "test answer",
        "expected_answer": None,
        "source_hint": "doc.pdf",
        "retriever_metrics": {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "avg_similarity": avg_sim,
            "retrieved_sources": ["doc.pdf"],
        },
        "generator_metrics": {
            "faithfulness": faithfulness,
            "faithfulness_reason": "good",
            "relevancy": relevancy,
            "relevancy_reason": "good",
            "correctness": correctness,
            "correctness_reason": "expected" if correctness is not None else "Эталонный ответ не задан",
        },
        "context_metrics": {
            "context_precision": context_precision,
            "context_precision_reason": "good",
            "context_recall": context_recall,
            "context_recall_reason": "good",
        },
        "latency_sec": latency,
    }


# ---------------------------------------------------------------------------
# compute_retriever_metrics
# ---------------------------------------------------------------------------


class TestComputeRetrieverMetrics:
    def test_with_matching_source_hint(self):
        from infrastructure.benchmark.evaluation.metrics import compute_retriever_metrics

        docs = [(_doc("text", source="report.pdf"), 0.9)]
        result = compute_retriever_metrics(docs, source_hint="report")
        assert result["hit_rate"] == 1
        assert result["mrr"] == 1.0
        assert result["avg_similarity"] == pytest.approx(0.9, abs=0.01)

    def test_with_no_match_source_hint(self):
        from infrastructure.benchmark.evaluation.metrics import compute_retriever_metrics

        docs = [(_doc("text", source="other.pdf"), 0.9)]
        result = compute_retriever_metrics(docs, source_hint="report")
        assert result["hit_rate"] == 0
        assert result["mrr"] == 0.0

    def test_with_none_source_hint(self):
        from infrastructure.benchmark.evaluation.metrics import compute_retriever_metrics

        docs = [(_doc("text"), 0.9)]
        result = compute_retriever_metrics(docs, source_hint=None)
        assert result["hit_rate"] is None
        assert result["mrr"] is None
        assert result["avg_similarity"] == pytest.approx(0.9, abs=0.01)

    def test_empty_docs(self):
        from infrastructure.benchmark.evaluation.metrics import compute_retriever_metrics

        result = compute_retriever_metrics([], source_hint="x")
        assert result["hit_rate"] == 0
        assert result["avg_similarity"] == 0.0

    def test_mrr_first_rank(self):
        from infrastructure.benchmark.evaluation.metrics import compute_retriever_metrics

        docs = [
            (_doc("a", source="target.pdf"), 0.9),
            (_doc("b", source="other.pdf"), 0.8),
        ]
        result = compute_retriever_metrics(docs, source_hint="target")
        assert result["mrr"] == 1.0

    def test_mrr_second_rank(self):
        from infrastructure.benchmark.evaluation.metrics import compute_retriever_metrics

        docs = [
            (_doc("a", source="other.pdf"), 0.9),
            (_doc("b", source="target.pdf"), 0.8),
        ]
        result = compute_retriever_metrics(docs, source_hint="target")
        assert result["mrr"] == pytest.approx(0.5, abs=0.01)

    def test_retrieved_sources_extracted(self):
        from infrastructure.benchmark.evaluation.metrics import compute_retriever_metrics

        docs = [(_doc("text", source="a.pdf"), 0.9), (_doc("text2", source="b.pdf"), 0.8)]
        result = compute_retriever_metrics(docs, source_hint=None)
        assert result["retrieved_sources"] == ["a.pdf", "b.pdf"]


# ---------------------------------------------------------------------------
# compute_summary_metrics
# ---------------------------------------------------------------------------


class TestComputeSummaryMetrics:
    def test_unknown_cost_does_not_become_zero_in_total(self):
        from infrastructure.benchmark.evaluation.metrics import compute_summary_metrics

        results = [{**_result(), 'cost_usd': 0.02}, {**_result(), 'cost_usd': None}]
        summary = compute_summary_metrics(results)
        assert summary['total_cost_usd'] is None
        assert summary['known_cost_usd'] == 0.02
        assert summary['cost_complete'] is False

    def test_basic_summary(self):
        from infrastructure.benchmark.evaluation.metrics import compute_summary_metrics

        results = [_result(), _result(faithfulness=6.0, relevancy=5.0)]
        summary = compute_summary_metrics(results)
        assert summary["total_questions"] == 2
        assert summary["avg_faithfulness"] == pytest.approx(7.0, abs=0.1)
        assert summary["avg_relevancy"] == pytest.approx(6.0, abs=0.1)

    def test_with_correctness(self):
        from infrastructure.benchmark.evaluation.metrics import compute_summary_metrics

        results = [_result(correctness=9.0), _result(correctness=7.0)]
        summary = compute_summary_metrics(results)
        assert summary["avg_correctness"] == pytest.approx(8.0, abs=0.1)

    def test_without_correctness(self):
        from infrastructure.benchmark.evaluation.metrics import compute_summary_metrics

        results = [_result(correctness=None)]
        summary = compute_summary_metrics(results)
        assert summary["avg_correctness"] is None

    def test_hit_rate_avg(self):
        from infrastructure.benchmark.evaluation.metrics import compute_summary_metrics

        results = [_result(hit_rate=1), _result(hit_rate=0)]
        summary = compute_summary_metrics(results)
        assert summary["hit_rate"] == pytest.approx(0.5, abs=0.01)

    def test_total_time(self):
        from infrastructure.benchmark.evaluation.metrics import compute_summary_metrics

        results = [_result(latency=1.0), _result(latency=2.0)]
        summary = compute_summary_metrics(results)
        assert summary["total_time_sec"] == pytest.approx(3.0, abs=0.1)

    def test_empty_results(self):
        from infrastructure.benchmark.evaluation.metrics import compute_summary_metrics

        summary = compute_summary_metrics([])
        assert summary["total_questions"] == 0

    def test_context_metrics(self):
        from infrastructure.benchmark.evaluation.metrics import compute_summary_metrics

        results = [_result(context_precision=8.0, context_recall=7.0)]
        summary = compute_summary_metrics(results)
        assert summary["avg_context_precision"] == pytest.approx(8.0, abs=0.1)
        assert summary["avg_context_recall"] == pytest.approx(7.0, abs=0.1)


# ---------------------------------------------------------------------------
# _extract_source_name
# ---------------------------------------------------------------------------


class TestExtractSourceName:
    def test_from_filename(self):
        from infrastructure.benchmark.evaluation.metrics import _extract_source_name

        doc = _doc("text", filename="report.pdf")
        assert _extract_source_name(doc) == "report.pdf"

    def test_from_source_path(self):
        from infrastructure.benchmark.evaluation.metrics import _extract_source_name

        doc = _doc("text", source="/path/to/report.pdf")
        assert _extract_source_name(doc) == "report.pdf"

    def test_no_metadata(self):
        from infrastructure.benchmark.evaluation.metrics import _extract_source_name

        doc = Document(page_content="text", metadata={})
        assert _extract_source_name(doc) == "?"


# ---------------------------------------------------------------------------
# _safe_avg
# ---------------------------------------------------------------------------


class TestSafeAvg:
    def test_normal(self):
        from infrastructure.benchmark.evaluation.metrics import _safe_avg

        assert _safe_avg([1.0, 2.0, 3.0]) == 2.0

    def test_empty(self):
        from infrastructure.benchmark.evaluation.metrics import _safe_avg

        assert _safe_avg([]) == 0

    def test_single(self):
        from infrastructure.benchmark.evaluation.metrics import _safe_avg

        assert _safe_avg([5.0]) == 5.0
