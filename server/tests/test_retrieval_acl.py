"""Tests for ACL filter propagation to Qdrant calls (F-017 regression).

Verifies that must-style filters (from with_temporal_filter / with_domain_filter)
are correctly passed to qdrant_dense_search and resolve_hashes_batch,
rather than being silently dropped.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from infrastructure.repositories.vector.acl import (  # noqa: E402
    build_qdrant_filter,
    with_domain_filter,
    with_temporal_filter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(kind="internal", uid=5):
    return {"id": uid, "kind": kind, "role": "user"}


def _make_ml_clients(qdrant_client=None, embeddings=None, bm25_index=None):
    ml = MagicMock()
    ml.qdrant_client.return_value = qdrant_client or MagicMock()
    embs = embeddings or MagicMock()
    embs.embed_query = AsyncMock(return_value=[0.1] * 384)
    ml.embeddings.return_value = embs
    ml._ensure_bm25_loaded = AsyncMock(return_value=bm25_index)
    return ml


# ---------------------------------------------------------------------------
# qdrant_dense_search: must-style filter must be passed to client.search
# ---------------------------------------------------------------------------

class TestDenseSearchACLFilter:
    def test_normal_acl_filter_passed_to_search(self):
        from infrastructure.ml.rag.rag_retrieval import qdrant_dense_search

        af = build_qdrant_filter(_user(), [1, 2])
        mock_client = MagicMock()
        mock_client.search.return_value = []

        ml = _make_ml_clients(qdrant_client=mock_client)
        asyncio.run(qdrant_dense_search("test", 10, af, ml))

        call_kwargs = mock_client.search.call_args
        assert call_kwargs.kwargs["query_filter"] is not None

    def test_must_style_filter_after_temporal_NOT_dropped(self):
        from infrastructure.ml.rag.rag_retrieval import qdrant_dense_search

        af = build_qdrant_filter(_user(), [1, 2])
        temporal_af = with_temporal_filter(af, None)

        mock_client = MagicMock()
        mock_client.search.return_value = []

        ml = _make_ml_clients(qdrant_client=mock_client)
        asyncio.run(qdrant_dense_search("test", 10, temporal_af, ml))

        call_kwargs = mock_client.search.call_args
        applied_filter = call_kwargs.kwargs["query_filter"]
        assert applied_filter is not None, (
            "Must-style filter from with_temporal_filter was dropped — "
            "dense search ran without ACL!"
        )
        assert applied_filter is temporal_af

    def test_must_style_filter_after_domain_NOT_dropped(self):
        from infrastructure.ml.rag.rag_retrieval import qdrant_dense_search

        af = build_qdrant_filter(_user(), [1, 2])
        legal_af = with_domain_filter(af, "legal")

        mock_client = MagicMock()
        mock_client.search.return_value = []

        ml = _make_ml_clients(qdrant_client=mock_client)
        asyncio.run(qdrant_dense_search("test", 10, legal_af, ml))

        call_kwargs = mock_client.search.call_args
        applied_filter = call_kwargs.kwargs["query_filter"]
        assert applied_filter is not None
        assert applied_filter is legal_af

    def test_none_filter_means_unfiltered(self):
        from infrastructure.ml.rag.rag_retrieval import qdrant_dense_search

        mock_client = MagicMock()
        mock_client.search.return_value = []

        ml = _make_ml_clients(qdrant_client=mock_client)
        asyncio.run(qdrant_dense_search("test", 10, None, ml))

        call_kwargs = mock_client.search.call_args
        assert call_kwargs.kwargs["query_filter"] is None


# ---------------------------------------------------------------------------
# resolve_hashes_batch: must-style filter must wrap scroll_filter
# ---------------------------------------------------------------------------

class TestResolveHashesACLFilter:
    def test_must_style_filter_includes_acl_in_scroll(self):
        from infrastructure.ml.rag.rag_retrieval import resolve_hashes_batch

        af = build_qdrant_filter(_user(), [1, 2])
        temporal_af = with_temporal_filter(af, None)
        # temporal_af = Filter(must=[...]) → should=None

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        ml = _make_ml_clients(qdrant_client=mock_client)
        asyncio.run(resolve_hashes_batch(["hash1", "hash2"], temporal_af, ml))

        call_kwargs = mock_client.scroll.call_args
        scroll_filter = call_kwargs.kwargs["scroll_filter"]
        assert scroll_filter is not None
        assert scroll_filter.must is not None
        assert temporal_af in scroll_filter.must

    def test_none_filter_means_unfiltered_scroll(self):
        from infrastructure.ml.rag.rag_retrieval import resolve_hashes_batch

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        ml = _make_ml_clients(qdrant_client=mock_client)
        asyncio.run(resolve_hashes_batch(["hash1"], None, ml))

        call_kwargs = mock_client.scroll.call_args
        scroll_filter = call_kwargs.kwargs["scroll_filter"]
        assert scroll_filter.must is None
