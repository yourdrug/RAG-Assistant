"""Regression: repeated 2.3 splits must not evict sparse-only provision 2.2."""

from unittest.mock import AsyncMock

import pytest
from langchain_core.documents import Document

from application.services.retrieval import HybridRetriever
from domain.services.retrieval_diversity import prioritize_distinct_provisions
from infrastructure.ml.rag.rag_reranking import rerank_documents


def provision(text, section, document_id=22, **metadata):
    return Document(page_content=text, metadata={"document_id": document_id, "section": section, **metadata})


def test_sparse_only_provision_survives_dense_repeated_section_at_same_budget():
    dense = [provision(f"daily {i}", "point 2 > subpoint_num 2.3 > sentence") for i in range(30)]
    target = provision("real time", "point 2 > subpoint_num 2.2")
    docs = {str(i): (0.8, doc) for i, doc in enumerate(dense)}
    docs["target"] = (0, target)
    candidates = HybridRetriever().merge_and_dedup(
        [(str(i), 0.8) for i in range(30)],
        [("target", 25.0)],
        docs,
        fetch_k=30,
        rrf_k=60,
        dense_weight=1.5,
        sparse_weight=0.5,
    )
    assert len(candidates) == 30
    assert candidates[:2] == [dense[0], target]


@pytest.mark.asyncio
async def test_reranker_retains_distinct_provision_without_changing_thresholds():
    repeated = [provision(f"daily {i}", "point 2 > subpoint_num 2.3 > sentence") for i in range(13)]
    target = provision("real time", "point 2 > subpoint_num 2.2")
    irrelevant = provision("unrelated", "point 3")
    reranker = AsyncMock()
    reranker.predict.return_value = [0.9998] * 13 + [0.9994, 0.01]
    ranked = await rerank_documents(
        "mode?", [*repeated, target, irrelevant], 10, reranker, min_score=0.15, score_gap_ratio=0.1
    )
    assert len(ranked) == 10
    assert ranked[:2] == [(repeated[0], 0.9998), (target, 0.9994)]
    assert irrelevant not in [doc for doc, _ in ranked]


def test_scope_tables_and_unstructured_chunks_are_not_collapsed():
    docs = [
        provision("a", "point 2", act_version_id=1),
        provision("b", "point 2", act_version_id=2),
        provision("c", "point 2", document_id=23),
        provision("d", "point 2", table_id="table-1"),
        provision("e", "point 2", table_id="table-1"),
        Document(page_content="unstructured"),
    ]
    assert prioritize_distinct_provisions(docs) == docs


def test_remaining_splits_fill_budget_in_score_order():
    docs = [provision(str(i), "point 2 > sentence") for i in range(3)]
    assert prioritize_distinct_provisions(docs) == docs
