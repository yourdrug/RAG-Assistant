"""Characterization tests for benchmark/benchmark.py — 1088-line god-file.

Locks down the pure-logic functions BEFORE refactoring.
LLM judge calls and Qdrant/retrieval are fully mocked.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        from infrastructure.benchmark.benchmark import compute_retriever_metrics

        docs = [(_doc("text", source="report.pdf"), 0.9)]
        result = compute_retriever_metrics(docs, source_hint="report")
        assert result["hit_rate"] == 1
        assert result["mrr"] == 1.0
        assert result["avg_similarity"] == pytest.approx(0.9, abs=0.01)

    def test_with_no_match_source_hint(self):
        from infrastructure.benchmark.benchmark import compute_retriever_metrics

        docs = [(_doc("text", source="other.pdf"), 0.9)]
        result = compute_retriever_metrics(docs, source_hint="report")
        assert result["hit_rate"] == 0
        assert result["mrr"] == 0.0

    def test_with_none_source_hint(self):
        from infrastructure.benchmark.benchmark import compute_retriever_metrics

        docs = [(_doc("text"), 0.9)]
        result = compute_retriever_metrics(docs, source_hint=None)
        assert result["hit_rate"] is None
        assert result["mrr"] is None
        assert result["avg_similarity"] == pytest.approx(0.9, abs=0.01)

    def test_empty_docs(self):
        from infrastructure.benchmark.benchmark import compute_retriever_metrics

        result = compute_retriever_metrics([], source_hint="x")
        assert result["hit_rate"] == 0
        assert result["avg_similarity"] == 0.0

    def test_mrr_first_rank(self):
        from infrastructure.benchmark.benchmark import compute_retriever_metrics

        docs = [
            (_doc("a", source="target.pdf"), 0.9),
            (_doc("b", source="other.pdf"), 0.8),
        ]
        result = compute_retriever_metrics(docs, source_hint="target")
        assert result["mrr"] == 1.0

    def test_mrr_second_rank(self):
        from infrastructure.benchmark.benchmark import compute_retriever_metrics

        docs = [
            (_doc("a", source="other.pdf"), 0.9),
            (_doc("b", source="target.pdf"), 0.8),
        ]
        result = compute_retriever_metrics(docs, source_hint="target")
        assert result["mrr"] == pytest.approx(0.5, abs=0.01)

    def test_retrieved_sources_extracted(self):
        from infrastructure.benchmark.benchmark import compute_retriever_metrics

        docs = [(_doc("text", source="a.pdf"), 0.9), (_doc("text2", source="b.pdf"), 0.8)]
        result = compute_retriever_metrics(docs, source_hint=None)
        assert result["retrieved_sources"] == ["a.pdf", "b.pdf"]


# ---------------------------------------------------------------------------
# compute_summary_metrics
# ---------------------------------------------------------------------------


class TestComputeSummaryMetrics:
    def test_basic_summary(self):
        from infrastructure.benchmark.benchmark import compute_summary_metrics

        results = [_result(), _result(faithfulness=6.0, relevancy=5.0)]
        summary = compute_summary_metrics(results)
        assert summary["total_questions"] == 2
        assert summary["avg_faithfulness"] == pytest.approx(7.0, abs=0.1)
        assert summary["avg_relevancy"] == pytest.approx(6.0, abs=0.1)

    def test_with_correctness(self):
        from infrastructure.benchmark.benchmark import compute_summary_metrics

        results = [_result(correctness=9.0), _result(correctness=7.0)]
        summary = compute_summary_metrics(results)
        assert summary["avg_correctness"] == pytest.approx(8.0, abs=0.1)

    def test_without_correctness(self):
        from infrastructure.benchmark.benchmark import compute_summary_metrics

        results = [_result(correctness=None)]
        summary = compute_summary_metrics(results)
        assert summary["avg_correctness"] is None

    def test_hit_rate_avg(self):
        from infrastructure.benchmark.benchmark import compute_summary_metrics

        results = [_result(hit_rate=1), _result(hit_rate=0)]
        summary = compute_summary_metrics(results)
        assert summary["hit_rate"] == pytest.approx(0.5, abs=0.01)

    def test_total_time(self):
        from infrastructure.benchmark.benchmark import compute_summary_metrics

        results = [_result(latency=1.0), _result(latency=2.0)]
        summary = compute_summary_metrics(results)
        assert summary["total_time_sec"] == pytest.approx(3.0, abs=0.1)

    def test_empty_results(self):
        from infrastructure.benchmark.benchmark import compute_summary_metrics

        summary = compute_summary_metrics([])
        assert summary["total_questions"] == 0

    def test_context_metrics(self):
        from infrastructure.benchmark.benchmark import compute_summary_metrics

        results = [_result(context_precision=8.0, context_recall=7.0)]
        summary = compute_summary_metrics(results)
        assert summary["avg_context_precision"] == pytest.approx(8.0, abs=0.1)
        assert summary["avg_context_recall"] == pytest.approx(7.0, abs=0.1)


# ---------------------------------------------------------------------------
# _extract_source_name
# ---------------------------------------------------------------------------


class TestExtractSourceName:
    def test_from_filename(self):
        from infrastructure.benchmark.benchmark import _extract_source_name

        doc = _doc("text", filename="report.pdf")
        assert _extract_source_name(doc) == "report.pdf"

    def test_from_source_path(self):
        from infrastructure.benchmark.benchmark import _extract_source_name

        doc = _doc("text", source="/path/to/report.pdf")
        assert _extract_source_name(doc) == "report.pdf"

    def test_no_metadata(self):
        from infrastructure.benchmark.benchmark import _extract_source_name

        doc = Document(page_content="text", metadata={})
        assert _extract_source_name(doc) == "?"


# ---------------------------------------------------------------------------
# _sanitize_model_name
# ---------------------------------------------------------------------------


class TestSanitizeModelName:
    def test_colon_replaced(self):
        from infrastructure.benchmark.benchmark import _sanitize_model_name

        assert _sanitize_model_name("model:v2") == "model_v2"

    def test_clean_name_unchanged(self):
        from infrastructure.benchmark.benchmark import _sanitize_model_name

        assert _sanitize_model_name("model-v2") == "model-v2"

    def test_special_chars(self):
        from infrastructure.benchmark.benchmark import _sanitize_model_name

        result = _sanitize_model_name('a/b:c*d?"e<f>g|h')
        assert "/" not in result
        assert ":" not in result
        assert "*" not in result


# ---------------------------------------------------------------------------
# _apply_rerank_filters
# ---------------------------------------------------------------------------


class TestApplyRerankFilters:
    def test_min_score_filters_low(self):
        from infrastructure.benchmark.benchmark import _apply_rerank_filters

        with patch("infrastructure.benchmark.retrieval.get_setting") as mock_get:
            mock_get.side_effect = lambda key: {
                "rag.rerank_min_score": 0.5,
                "rag.rerank_score_gap_ratio": None,
            }.get(key)
            docs = [
                (_doc("a"), 0.9),
                (_doc("b"), 0.3),
                (_doc("c"), 0.7),
            ]
            result = _apply_rerank_filters(docs)
            assert len(result) == 2
            assert all(s >= 0.5 for _, s in result)

    def test_gap_ratio_filters(self):
        from infrastructure.benchmark.benchmark import _apply_rerank_filters

        with patch("infrastructure.benchmark.retrieval.get_setting") as mock_get:
            mock_get.side_effect = lambda key: {
                "rag.rerank_min_score": None,
                "rag.rerank_score_gap_ratio": 0.5,
            }.get(key)
            docs = [
                (_doc("a"), 1.0),
                (_doc("b"), 0.6),
                (_doc("c"), 0.3),
            ]
            result = _apply_rerank_filters(docs)
            # cutoff = 1.0 * 0.5 = 0.5, so 0.3 is filtered
            assert len(result) == 2

    def test_no_filters(self):
        from infrastructure.benchmark.benchmark import _apply_rerank_filters

        with patch("infrastructure.benchmark.retrieval.get_setting", return_value=None):
            docs = [(_doc("a"), 0.9), (_doc("b"), 0.1)]
            result = _apply_rerank_filters(docs)
            assert len(result) == 2


# ---------------------------------------------------------------------------
# save_results (filesystem)
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_creates_json_and_csv(self, tmp_path):
        from infrastructure.benchmark.benchmark import save_results

        results = [_result()]
        out_dir = str(tmp_path / "bench_out")
        with patch("infrastructure.benchmark.persistence.settings") as mock_settings:
            mock_settings.retriever_top_k = 4
            mock_settings.chunk_size = 500
            mock_settings.chunk_overlap = 50
            mock_settings.tei_embed_url = "http://localhost:8080"
            mock_settings.llm_model = "test-model"
            mock_settings.hybrid_enabled = False
            mock_settings.dense_weight = 1.0
            mock_settings.sparse_weight = 1.0
            mock_settings.rrf_k = 30
            mock_settings.data_dir = str(tmp_path)
            with patch("infrastructure.benchmark.persistence.save_summary_to_history"):
                save_results(results, out_dir, model_name="test-model")

        out = Path(out_dir)
        json_files = list(out.glob("benchmark_*.json"))
        csv_files = list(out.glob("benchmark_*.csv"))
        assert len(json_files) == 1
        assert len(csv_files) == 1

        data = json.loads(json_files[0].read_text())
        assert len(data) == 1
        assert data[0]["id"] == "q1"

    def test_csv_has_header(self, tmp_path):
        from infrastructure.benchmark.benchmark import save_results

        results = [_result()]
        out_dir = str(tmp_path / "bench_csv")
        with patch("infrastructure.benchmark.persistence.settings") as mock_settings:
            mock_settings.retriever_top_k = 4
            mock_settings.chunk_size = 500
            mock_settings.chunk_overlap = 50
            mock_settings.tei_embed_url = "http://localhost:8080"
            mock_settings.llm_model = "m"
            mock_settings.hybrid_enabled = False
            mock_settings.dense_weight = 1.0
            mock_settings.sparse_weight = 1.0
            mock_settings.rrf_k = 30
            mock_settings.data_dir = str(tmp_path)
            with patch("infrastructure.benchmark.persistence.save_summary_to_history"):
                save_results(results, out_dir)

        csv_path = list(Path(out_dir).glob("benchmark_*.csv"))[0]
        lines = csv_path.read_text().split("\n")
        assert lines[0].startswith("id,question,faithfulness")


# ---------------------------------------------------------------------------
# load_questions
# ---------------------------------------------------------------------------


class TestLoadQuestions:
    def test_loads_json_file(self, tmp_path):
        from infrastructure.benchmark.benchmark import load_questions

        q_file = tmp_path / "questions.json"
        q_file.write_text(json.dumps([{"id": "q1", "question": "test?"}]))
        result = load_questions(str(q_file))
        assert len(result) == 1
        assert result[0]["id"] == "q1"


# ---------------------------------------------------------------------------
# _safe_avg
# ---------------------------------------------------------------------------


class TestSafeAvg:
    def test_normal(self):
        from infrastructure.benchmark.benchmark import _safe_avg

        assert _safe_avg([1.0, 2.0, 3.0]) == 2.0

    def test_empty(self):
        from infrastructure.benchmark.benchmark import _safe_avg

        assert _safe_avg([]) == 0

    def test_single(self):
        from infrastructure.benchmark.benchmark import _safe_avg

        assert _safe_avg([5.0]) == 5.0


# ---------------------------------------------------------------------------
# get_rag_answer (mocked LLM)
# ---------------------------------------------------------------------------


class TestGetRagAnswer:
    def test_returns_llm_response(self):
        from infrastructure.benchmark.benchmark import get_rag_answer

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "The answer is 42."
        mock_llm.invoke.return_value = mock_response

        docs = [(_doc("context text"), 0.9)]
        result = get_rag_answer(mock_llm, docs, "What is the answer?")
        assert result == "The answer is 42."

    def test_retries_on_exception(self):
        from infrastructure.benchmark.benchmark import get_rag_answer

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "ok"
        mock_llm.invoke.side_effect = [Exception("fail"), mock_response]

        docs = [(_doc("text"), 0.9)]
        result = get_rag_answer(mock_llm, docs, "q")
        assert result == "ok"
        assert mock_llm.invoke.call_count == 2
