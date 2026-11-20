"""The offline retrieval proxy fuses and reranks like production retrieval."""

from __future__ import annotations

from pathlib import Path
import runpy
from typing import Any

import pytest

from config.sections.rag import HybridSearchSection, RankingSection, RetrievalSection
from domain.rag.classification import has_exact_reference


@pytest.fixture(scope="module")
def report_tool() -> dict[str, Any]:
    script = Path(__file__).resolve().parents[2] / "scripts/benchmark/chunking_report.py"
    return runpy.run_path(str(script))


def test_proxy_uses_production_retrieval_defaults(report_tool: dict[str, Any]) -> None:
    assert report_tool["RRF_K"] == HybridSearchSection().rrf_k
    assert report_tool["SPARSE_DEPTH"] == HybridSearchSection().bm25_fetch_k
    assert report_tool["DENSE_DEPTH"] == RetrievalSection().retriever_fetch_k
    assert report_tool["RERANK_TOP_N"] == RetrievalSection().retriever_top_k_broad


def test_rrf_boosts_sparse_only_for_exact_references(report_tool: dict[str, Any]) -> None:
    plain, exact = "Как оформить отпуск?", "Что сказано в статье 5?"
    assert not has_exact_reference(plain)
    assert has_exact_reference(exact)

    # Equal weights tie the mirrored lists; the dense leader stays first.
    assert report_tool["rrf"](plain, [0, 1], [1, 0]) == [0, 1]
    # The boosted sparse list wins for an exact structural reference.
    assert report_tool["rrf"](exact, [0, 1], [1, 0]) == [1, 0]


def test_rrf_keeps_fused_depth(report_tool: dict[str, Any]) -> None:
    depth = report_tool["FUSED_DEPTH"]
    fused = report_tool["rrf"]("вопрос", list(range(depth)), list(range(depth, 2 * depth)))
    assert len(fused) == depth


class FakeRerankCache:
    def __init__(self, scores: dict[str, float]) -> None:
        self.by_text = scores

    def scores(self, _query: str, texts: list[str]) -> list[float]:
        return [self.by_text[text] for text in texts]


def test_rerank_reorders_and_applies_production_filters(report_tool: dict[str, Any]) -> None:
    ranking = RankingSection()
    top_n = report_tool["RERANK_TOP_N"]
    low = (ranking.rerank_min_score or 0.0) / 2
    chunks = [{"rerank_text": f"chunk {i}"} for i in range(top_n + 3)]
    scores = {f"chunk {i}": 0.5 + i / 100 for i in range(top_n + 2)}
    scores[f"chunk {top_n + 2}"] = low
    question = {"case_id": "q1", "question": "вопрос"}

    ranked = report_tool["rerank_rankings"](
        [question], {"q1": list(range(top_n + 3))}, chunks, FakeRerankCache(scores)
    )["q1"]

    assert ranked == list(range(top_n + 1, 1, -1))
    assert top_n + 2 not in ranked
