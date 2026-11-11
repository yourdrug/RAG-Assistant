"""Characterization tests for infrastructure.ml.rag.rag_retrieval.

Tests the retrieval primitives (dense search, hash resolution, hybrid search)
with mocked Qdrant/BM25/embeddings to verify correct orchestration logic.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue, NamedSparseVector

from domain.knowledge.access_control import VisibilityCondition
from domain.knowledge.enums import DocumentVisibility
from domain.rag.settings import (
    FeatureToggles,
    HybridSearchConfig,
    RagSettings,
    RerankConfig,
    RetrieverConfig,
)
from infrastructure.ml.rag.rag_retrieval import (
    qdrant_dense_search,
    query_vector_scope,
    resolve_hashes_batch,
    run_hybrid_search,
)
from infrastructure.repositories.vector.acl import build_qdrant_filter
from tests.test_rag_pipeline import make_rag


# ---------------------------------------------------------------------------
# resolve_hashes_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_hashes_batch_empty_returns_empty() -> None:

    result = await resolve_hashes_batch([], None, MagicMock())
    assert result == {}


@pytest.mark.asyncio
@patch("config.settings")
async def test_resolve_hashes_batch_calls_scroll(mock_settings: MagicMock) -> None:

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
@patch("config.settings")
async def test_resolve_hashes_batch_with_access_filter(mock_settings: MagicMock) -> None:

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
@patch("config.settings")
async def test_qdrant_dense_search_returns_hash_score_doc_tuples(mock_settings: MagicMock) -> None:

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

    results = await qdrant_dense_search(
        "test query", 5, build_qdrant_filter({"id": 10, "kind": "internal", "role": "user"}, []), ml_clients
    )

    assert len(results) == 1
    h, score, doc = results[0]
    assert h == "hash1"
    assert score == 0.85
    assert doc.page_content == "chunk text"
    mock_embeddings.embed_query.assert_awaited_once_with("test query")


@pytest.mark.asyncio
@patch("config.settings")
async def test_qdrant_dense_search_generates_hash_when_missing(mock_settings: MagicMock) -> None:

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

    results = await qdrant_dense_search(
        "query", 3, build_qdrant_filter({"id": 10, "kind": "internal", "role": "user"}, []), ml_clients
    )

    h, _score, _doc = results[0]
    assert h is not None  # hash was generated from content
    assert len(h) > 0


# ---------------------------------------------------------------------------
# run_hybrid_search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("infrastructure.ml.rag.rag_retrieval.RAG_STAGE_DURATION")
@patch("config.settings")
async def test_hybrid_search_fallback_to_dense_when_bm25_disabled(
    mock_settings: MagicMock, mock_metrics: MagicMock
) -> None:

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
    ml_clients.ensure_bm25_loaded = AsyncMock(return_value=None)  # no BM25

    _acl_filter = Filter(must=[FieldCondition(key="id", match=MatchValue(value=0))])

    results = await run_hybrid_search(
        "test",
        10,
        _acl_filter,
        rag,
        ml_clients=ml_clients,
        visibility_conditions=[VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PUBLIC)],
        user_id=10,
        user_group_ids=[],
    )

    assert len(results) >= 1
    assert results[0].page_content == "result"


@pytest.mark.asyncio
@patch("infrastructure.ml.rag.rag_retrieval.RAG_STAGE_DURATION")
@patch("config.settings")
async def test_run_hybrid_search_acl_defaults_are_mandatory(
    mock_settings: MagicMock, mock_metrics: MagicMock
) -> None:

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

    ml_clients = MagicMock()
    ml_clients.ensure_bm25_loaded = AsyncMock(return_value=None)

    with pytest.raises(RuntimeError, match="requires access_filter"):
        await run_hybrid_search(
            "test",
            10,
            None,
            rag,
            ml_clients=ml_clients,
            visibility_conditions=[],
            user_id=10,
            user_group_ids=[],
        )

    for missing in ("visibility_conditions", "user_id", "user_group_ids"):
        conditions: list[VisibilityCondition] | None = [] if missing != "visibility_conditions" else None
        uid = 10 if missing != "user_id" else None
        groups: list[int] | None = [] if missing != "user_group_ids" else None
        with pytest.raises(RuntimeError, match="requires"):
            await run_hybrid_search(
                "test",
                10,
                Filter(must=[]),
                rag,
                ml_clients=ml_clients,
                visibility_conditions=conditions,
                user_id=uid,
                user_group_ids=groups,
            )

    ml_clients.ensure_bm25_loaded.assert_not_awaited()


@pytest.mark.asyncio
@patch("infrastructure.ml.rag.rag_retrieval.RAG_STAGE_DURATION")
@patch("config.settings")
async def test_run_hybrid_search_empty_conditions_fail_closed(
    mock_settings: MagicMock, mock_metrics: MagicMock
) -> None:

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

    mock_client = MagicMock()
    mock_client.search.return_value = []

    mock_embeddings = MagicMock()
    mock_embeddings.embed_query = AsyncMock(return_value=[0.1] * 384)

    ml_clients = MagicMock()
    ml_clients.qdrant_client.return_value = mock_client
    ml_clients.embeddings.return_value = mock_embeddings
    ml_clients.ensure_bm25_loaded = AsyncMock(return_value=MagicMock())

    results = await run_hybrid_search(
        "test",
        10,
        Filter(must=[FieldCondition(key="id", match=MatchValue(value=0))]),
        rag,
        ml_clients=ml_clients,
        visibility_conditions=[],
        user_id=10,
        user_group_ids=[],
    )

    assert results == []
    mock_client.search.assert_not_called()
    ml_clients.embeddings.assert_not_called()
    ml_clients.embeddings.return_value.embed_query.assert_not_awaited()
    ml_clients.ensure_bm25_loaded.assert_not_awaited()


@pytest.mark.asyncio
@patch("application.search.hybrid.rrf_merge")
@patch("infrastructure.ml.rag.rag_retrieval.RAG_STAGE_DURATION")
@patch("config.settings")
async def test_hybrid_search_uses_rrf_with_qdrant_sparse(
    mock_settings: MagicMock, mock_metrics: MagicMock, mock_rrf: MagicMock
) -> None:

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
    dense_point.id = 1
    dense_point.score = 0.85
    dense_point.payload = {
        "page_content": "dense doc",
        "metadata": {"content_hash": "dense_h1"},
    }
    mock_client = MagicMock()
    sparse_point = MagicMock(id=2, score=1.2)
    sparse_point.payload = {"page_content": "sparse doc", "metadata": {"chunk_id": 2}}
    mock_client.search.side_effect = lambda **kwargs: (
        [sparse_point] if isinstance(kwargs["query_vector"], NamedSparseVector) else [dense_point]
    )

    mock_embeddings = MagicMock()
    mock_embeddings.embed_query = AsyncMock(return_value=[0.1] * 384)

    # RRF merge returns dense hash first
    mock_rrf.return_value = ["1", "2"]

    ml_clients = MagicMock()
    ml_clients.qdrant_client.return_value = mock_client
    ml_clients.embeddings.return_value = mock_embeddings
    ml_clients.ensure_bm25_loaded = AsyncMock(side_effect=AssertionError("obsolete snapshot load"))

    _acl_filter = Filter(must=[FieldCondition(key="id", match=MatchValue(value=0))])
    conditions = [VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PUBLIC)]
    results = await run_hybrid_search(
        "test",
        10,
        _acl_filter,
        rag,
        ml_clients=ml_clients,
        visibility_conditions=conditions,
        user_id=10,
        user_group_ids=[],
    )

    assert mock_client.search.call_count == 2
    assert all(call.kwargs["query_filter"] is _acl_filter for call in mock_client.search.call_args_list)
    mock_client.scroll.assert_not_called()
    ml_clients.ensure_bm25_loaded.assert_not_awaited()
    mock_rrf.assert_called_once()
    assert len(results) >= 1


@pytest.fixture(autouse=True)
def mock_embedding_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "infrastructure.ml.rag.rag_retrieval.ensure_embedding_identity", lambda *_args: "test:model:v1"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("programming error"), asyncio.CancelledError()])
async def test_sparse_errors_and_cancellation_propagate(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:

    monkeypatch.setattr(
        "infrastructure.ml.rag.rag_retrieval.qdrant_sparse_search", AsyncMock(side_effect=error)
    )
    monkeypatch.setattr("infrastructure.ml.rag.rag_retrieval.qdrant_dense_search", AsyncMock(return_value=[]))
    with pytest.raises(type(error)):
        await run_hybrid_search(
            "query",
            10,
            Filter(must=[]),
            make_rag(),
            ml_clients=MagicMock(),
            visibility_conditions=[VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PUBLIC)],
            user_id=42,
            user_group_ids=[],
        )


@pytest.mark.asyncio
@patch("config.settings")
async def test_dense_search_reuses_query_vector_within_request_scope(mock_settings: MagicMock) -> None:
    mock_settings.collection_name = "test_col"
    mock_settings.qdrant_timeout = 10
    mock_client = MagicMock()
    mock_client.search.return_value = []
    mock_embeddings = MagicMock()
    mock_embeddings.embedding_identity = "test:model:v1"
    mock_embeddings.embed_query = AsyncMock(return_value=[0.1] * 384)
    ml_clients = MagicMock()
    ml_clients.qdrant_client.return_value = mock_client
    ml_clients.embeddings.return_value = mock_embeddings
    acl = build_qdrant_filter({"id": 10, "kind": "internal", "role": "user"}, [])

    with query_vector_scope():
        await qdrant_dense_search("same query", 5, acl, ml_clients)
        await qdrant_dense_search("same query", 5, acl, ml_clients)
        await qdrant_dense_search("other query", 5, acl, ml_clients)

    assert [call.args[0] for call in mock_embeddings.embed_query.await_args_list] == [
        "same query",
        "other query",
    ]
    assert mock_client.search.call_count == 3

    # Outside a request scope nothing is memoized across calls.
    await qdrant_dense_search("same query", 5, acl, ml_clients)
    assert mock_embeddings.embed_query.await_count == 3
