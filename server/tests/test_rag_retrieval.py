"""Characterization tests for infrastructure.ml.rag.rag_retrieval.

Tests the retrieval primitives (dense search, hash resolution, hybrid search)
with mocked Qdrant/BM25/embeddings to verify correct orchestration logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue


# ---------------------------------------------------------------------------
# resolve_hashes_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_hashes_batch_empty_returns_empty():
    from infrastructure.ml.rag.rag_retrieval import resolve_hashes_batch

    result = await resolve_hashes_batch([], None, MagicMock())
    assert result == {}


@pytest.mark.asyncio
@patch("infrastructure.ml.rag.rag_retrieval.settings")
async def test_resolve_hashes_batch_calls_scroll(mock_settings):
    from infrastructure.ml.rag.rag_retrieval import resolve_hashes_batch

    mock_settings.collection_name = "test_col"
    mock_settings.qdrant_timeout = 10

    mock_client = MagicMock()
    point = MagicMock()
    point.payload = {
        "page_content": "hello world",
        "metadata": {"content_hash": "abc123", "source": "test.pdf"},
    }
    mock_client.scroll.return_value = ([point], None)

    ml_clients = MagicMock()
    ml_clients.qdrant_client.return_value = mock_client

    result = await resolve_hashes_batch(
        ["abc123"], Filter(must=[FieldCondition(key="id", match=MatchValue(value=0))]), ml_clients
    )

    assert "abc123" in result
    assert result["abc123"].page_content == "hello world"
    assert result["abc123"].metadata["source"] == "test.pdf"
    mock_client.scroll.assert_called_once()


@pytest.mark.asyncio
@patch("infrastructure.ml.rag.rag_retrieval.settings")
async def test_resolve_hashes_batch_with_access_filter(mock_settings):
    from infrastructure.ml.rag.rag_retrieval import resolve_hashes_batch

    mock_settings.collection_name = "test_col"
    mock_settings.qdrant_timeout = 10

    mock_client = MagicMock()
    point = MagicMock()
    point.payload = {
        "page_content": "doc",
        "metadata": {"content_hash": "h1"},
    }
    mock_client.scroll.return_value = ([point], None)

    ml_clients = MagicMock()
    ml_clients.qdrant_client.return_value = mock_client

    # Use a real Qdrant Filter with a Should condition
    access_filter = Filter(should=[FieldCondition(key="owner_id", match=MatchValue(value=42))])

    result = await resolve_hashes_batch(["h1"], access_filter, ml_clients)

    assert "h1" in result
    mock_client.scroll.assert_called_once()


# ---------------------------------------------------------------------------
# qdrant_dense_search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("infrastructure.ml.rag.rag_retrieval.settings")
async def test_qdrant_dense_search_returns_hash_score_doc_tuples(mock_settings):
    from infrastructure.ml.rag.rag_retrieval import qdrant_dense_search

    mock_settings.collection_name = "test_col"
    mock_settings.qdrant_timeout = 10

    point = MagicMock()
    point.score = 0.85
    point.payload = {
        "page_content": "chunk text",
        "metadata": {"content_hash": "hash1", "source": "doc.pdf"},
    }

    mock_client = MagicMock()
    mock_client.search.return_value = [point]

    mock_embeddings = MagicMock()
    mock_embeddings.embed_query = AsyncMock(return_value=[0.1] * 384)

    ml_clients = MagicMock()
    ml_clients.qdrant_client.return_value = mock_client
    ml_clients.embeddings.return_value = mock_embeddings

    results = await qdrant_dense_search("test query", 5, None, ml_clients)

    assert len(results) == 1
    h, score, doc = results[0]
    assert h == "hash1"
    assert score == 0.85
    assert doc.page_content == "chunk text"
    mock_embeddings.embed_query.assert_awaited_once_with("test query")


@pytest.mark.asyncio
@patch("infrastructure.ml.rag.rag_retrieval.settings")
async def test_qdrant_dense_search_generates_hash_when_missing(mock_settings):
    from infrastructure.ml.rag.rag_retrieval import qdrant_dense_search

    mock_settings.collection_name = "test_col"
    mock_settings.qdrant_timeout = 10

    point = MagicMock()
    point.score = 0.7
    point.payload = {
        "page_content": "some content",
        "metadata": {},  # no content_hash
    }

    mock_client = MagicMock()
    mock_client.search.return_value = [point]

    mock_embeddings = MagicMock()
    mock_embeddings.embed_query = AsyncMock(return_value=[0.1] * 384)

    ml_clients = MagicMock()
    ml_clients.qdrant_client.return_value = mock_client
    ml_clients.embeddings.return_value = mock_embeddings

    results = await qdrant_dense_search("query", 3, None, ml_clients)

    h, score, doc = results[0]
    assert h is not None  # hash was generated from content
    assert len(h) > 0


# ---------------------------------------------------------------------------
# run_hybrid_search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("infrastructure.ml.rag.rag_retrieval.RAG_STAGE_DURATION")
@patch("infrastructure.ml.rag.rag_retrieval.settings")
async def test_hybrid_search_fallback_to_dense_when_bm25_disabled(mock_settings, mock_metrics):
    from domain.value_objects.rag_settings import (
        FeatureToggles,
        HybridSearchConfig,
        RagSettings,
        RerankConfig,
        RetrieverConfig,
    )
    from infrastructure.ml.rag.rag_retrieval import run_hybrid_search

    mock_settings.collection_name = "test_col"
    mock_settings.qdrant_timeout = 10

    rag = RagSettings(
        retriever=RetrieverConfig(fetch_k=20, top_k=5, fetch_k_broad=40, top_k_broad=10),
        hybrid_search=HybridSearchConfig(
            enabled=True, bm25_fetch_k=20, rrf_k=60, dense_weight=1.0, sparse_weight=1.0
        ),
        rerank=RerankConfig(min_score=None, score_gap_ratio=None),
        features=FeatureToggles(
            citation_filter_enabled=False,
            relevance_gate_enabled=False,
            condense_enabled=False,
            decomposition_enabled=False,
            rolling_summary_enabled=False,
            cache_enabled=False,
        ),
        source_min_score=0.0,
    )

    point = MagicMock()
    point.score = 0.9
    point.payload = {
        "page_content": "result",
        "metadata": {"content_hash": "h1"},
    }

    mock_client = MagicMock()
    mock_client.search.return_value = [point]

    mock_embeddings = MagicMock()
    mock_embeddings.embed_query = AsyncMock(return_value=[0.1] * 384)

    ml_clients = MagicMock()
    ml_clients.qdrant_client.return_value = mock_client
    ml_clients.embeddings.return_value = mock_embeddings
    ml_clients._ensure_bm25_loaded = AsyncMock(return_value=None)  # no BM25

    _acl_filter = Filter(must=[FieldCondition(key="id", match=MatchValue(value=0))])

    results = await run_hybrid_search("test", 10, _acl_filter, rag, ml_clients)

    assert len(results) >= 1
    assert results[0].page_content == "result"


@pytest.mark.asyncio
@patch("application.services.retrieval.rrf_merge")
@patch("infrastructure.ml.rag.rag_retrieval.RAG_STAGE_DURATION")
@patch("infrastructure.ml.rag.rag_retrieval.settings")
async def test_hybrid_search_uses_rrf_when_bm25_available(mock_settings, mock_metrics, mock_rrf):
    from domain.value_objects.rag_settings import (
        FeatureToggles,
        HybridSearchConfig,
        RagSettings,
        RerankConfig,
        RetrieverConfig,
    )
    from infrastructure.ml.rag.rag_retrieval import run_hybrid_search

    mock_settings.collection_name = "test_col"
    mock_settings.qdrant_timeout = 10

    rag = RagSettings(
        retriever=RetrieverConfig(fetch_k=20, top_k=5, fetch_k_broad=40, top_k_broad=10),
        hybrid_search=HybridSearchConfig(
            enabled=True, bm25_fetch_k=20, rrf_k=60, dense_weight=1.0, sparse_weight=1.0
        ),
        rerank=RerankConfig(min_score=None, score_gap_ratio=None),
        features=FeatureToggles(
            citation_filter_enabled=False,
            relevance_gate_enabled=False,
            condense_enabled=False,
            decomposition_enabled=False,
            rolling_summary_enabled=False,
            cache_enabled=False,
        ),
        source_min_score=0.0,
    )

    # Dense results
    dense_point = MagicMock()
    dense_point.score = 0.85
    dense_point.payload = {
        "page_content": "dense doc",
        "metadata": {"content_hash": "dense_h1"},
    }
    mock_client = MagicMock()
    mock_client.search.return_value = [dense_point]

    mock_embeddings = MagicMock()
    mock_embeddings.embed_query = AsyncMock(return_value=[0.1] * 384)

    # BM25 results
    mock_bm25 = MagicMock()
    mock_bm25.search_with_hashes.return_value = [("sparse_h1", 0.7)]

    # RRF merge returns dense hash first
    mock_rrf.return_value = ["dense_h1", "sparse_h1"]

    ml_clients = MagicMock()
    ml_clients.qdrant_client.return_value = mock_client
    ml_clients.embeddings.return_value = mock_embeddings
    ml_clients._ensure_bm25_loaded = AsyncMock(return_value=mock_bm25)

    _acl_filter = Filter(must=[FieldCondition(key="id", match=MatchValue(value=0))])
    results = await run_hybrid_search("test", 10, _acl_filter, rag, ml_clients)

    mock_rrf.assert_called_once()
    assert len(results) >= 1
