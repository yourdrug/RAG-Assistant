"""Sweep scoring must honor the exclusions made while caching reranker scores."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain.schema import Document
import pytest

from infrastructure.benchmark.sweep.reranking import cache_reranker_scores
from infrastructure.benchmark.sweep.scoring import score_config_cheap
from infrastructure.benchmark.sweep.settings import settings_snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("sparse_only", [False, True])
@pytest.mark.parametrize("include_matching", [False, True])
async def test_phase_a_honors_query_scope_exclusions(sparse_only: bool, include_matching: bool) -> None:
    query = "Как часто передавать сведения о товарах, не включённых в перечень?"
    conflicting = Document(
        page_content="Сведения о товарах, включённых в перечень, передаются ежедневно.",
        metadata={"source": "wrong.pdf"},
    )
    matching = Document(
        page_content="Сведения о товарах, не включённых в перечень, передаются ежемесячно.",
        metadata={"source": "law.pdf"},
    )
    candidates = {"conflicting": conflicting}

    if include_matching:
        candidates["matching"] = matching

    dense = {} if sparse_only else {query: [(h, 1.0, doc) for h, doc in candidates.items()]}
    sparse = {query: [(h, 1.0) for h in candidates]} if sparse_only else {}
    questions = [{"question": query, "source_hint": "law.pdf"}]
    predict = AsyncMock(return_value=[0.9])
    scores = await cache_reranker_scores(
        questions, dense, sparse, candidates, SimpleNamespace(predict=predict)
    )

    assert scores[query] == ({"matching": 0.9} if include_matching else {})
    assert predict.await_count == int(include_matching)
    metrics = score_config_cheap(
        {"fetch_k": 2, "top_k": 1, "rerank_min_score": 0.5, "rerank_score_gap_ratio": None},
        questions,
        dense,
        sparse,
        candidates,
        weights={"hit_rate": 1},
        rerank_scores=scores,
        defaults=settings_snapshot(),
    )

    assert metrics["avg_hit_rate"] == int(include_matching)
    assert metrics["avg_mrr"] == int(include_matching)


def test_phase_a_does_not_hide_missing_scores_for_eligible_candidates() -> None:
    doc = Document(page_content="Relevant evidence", metadata={"source": "law.pdf"})
    with pytest.raises(KeyError, match="eligible"):
        score_config_cheap(
            {"fetch_k": 1, "top_k": 1},
            [{"question": "question", "source_hint": "law.pdf"}],
            {"question": [("eligible", 1.0, doc)]},
            {},
            {"eligible": doc},
            weights={"hit_rate": 1},
            rerank_scores={"question": {}},
            defaults=settings_snapshot(),
        )
