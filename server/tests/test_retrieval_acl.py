"""Tests for ACL filter propagation to Qdrant calls (F-017 regression).

Verifies that must-style filters (from with_temporal_filter / with_domain_filter)
are correctly passed to qdrant_dense_search and resolve_hashes_batch,
rather than being silently dropped.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest  # noqa: E402

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
        assert (
            applied_filter is not None
        ), "Must-style filter from with_temporal_filter was dropped — dense search ran without ACL!"
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

    def test_none_filter_rejected_before_io(self):
        from infrastructure.ml.rag.rag_retrieval import qdrant_dense_search

        mock_client = MagicMock()
        mock_client.search.return_value = []

        ml = _make_ml_clients(qdrant_client=mock_client)
        with pytest.raises(RuntimeError, match="requires access_filter"):
            asyncio.run(qdrant_dense_search("test", 10, None, ml))

        ml.qdrant_client.assert_not_called()
        ml.embeddings.assert_not_called()
        ml.embeddings.return_value.embed_query.assert_not_awaited()
        mock_client.search.assert_not_called()


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

    def test_none_filter_raises_runtime_error(self):
        """access_filter=None must raise RuntimeError (FINDING-020 security guard)."""
        from infrastructure.ml.rag.rag_retrieval import resolve_hashes_batch

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        ml = _make_ml_clients(qdrant_client=mock_client)
        with pytest.raises(RuntimeError, match="requires access_filter"):
            asyncio.run(resolve_hashes_batch(["hash1"], None, ml))


@pytest.mark.asyncio
@pytest.mark.parametrize("decomposition", [False, True])
@pytest.mark.parametrize("sub_queries", [["query"], ["first query", "second query"]])
async def test_decomposition_and_legal_fallback_preserve_acl(decomposition, sub_queries):
    from domain.services.access_control import get_visibility_conditions
    from domain.value_objects.llm_provider import Breadth
    from infrastructure.ml.rag.helpers import retrieve_with_decomposition

    af = with_temporal_filter(build_qdrant_filter(_user(), [1, 2]), None)
    conditions = get_visibility_conditions("internal", 5, [1, 2], for_list=False)
    rag = MagicMock()
    rag.features.decomposition_enabled = decomposition
    ml = _make_ml_clients()
    with (
        patch("infrastructure.ml.rag.helpers.decompose_question", AsyncMock(return_value=sub_queries)),
        patch("infrastructure.ml.rag.helpers.run_hybrid_search", AsyncMock(return_value=[])) as search,
    ):
        candidates, actual_queries = await retrieve_with_decomposition(
            "query",
            10,
            af,
            rag,
            ml,
            Breadth.NARROW,
            "legal",
            1.0,
            1.0,
            visibility_conditions=conditions,
            user_id=5,
            user_group_ids=[1, 2],
        )

    assert candidates == []
    assert actual_queries == (sub_queries if decomposition else ["query"])
    assert search.await_count == 2 * len(actual_queries)
    for domain_call, fallback_call in zip(
        search.await_args_list[::2],
        search.await_args_list[1::2],
        strict=True,
    ):
        assert af in domain_call.args[2].must
        assert fallback_call.args[2] is af
        for call in (domain_call, fallback_call):
            assert call.kwargs["visibility_conditions"] is conditions
            assert call.kwargs["user_id"] == 5
            assert call.kwargs["user_group_ids"] == [1, 2]


@pytest.mark.asyncio
async def test_legal_rerank_fallback_preserves_acl_and_temporal_filter():
    from domain.services.access_control import get_visibility_conditions
    from domain.value_objects.llm_provider import Breadth
    from domain.value_objects.chat_context import ChatContext
    from domain.value_objects.roles import UserKind, UserRole
    from infrastructure.ml.rag.helpers import rerank_and_enrich

    af = with_temporal_filter(build_qdrant_filter(_user(), [1, 2]), None)
    conditions = get_visibility_conditions("internal", 5, [1, 2], for_list=False)
    ctx = ChatContext(
        as_of_date=None,
        user_id=5,
        user_kind=UserKind.INTERNAL,
        user_role=UserRole.USER,
        user_group_ids=[1, 2],
    )
    rag = MagicMock()
    ml = _make_ml_clients()
    with (
        patch("infrastructure.ml.rag.helpers.rerank_documents", AsyncMock(return_value=[])),
        patch("infrastructure.ml.rag.helpers.run_hybrid_search", AsyncMock(return_value=[])) as search,
        patch("infrastructure.ml.rag.helpers.enrich_with_neighbors", AsyncMock(return_value=[])),
        patch("domain.services.rag_policy.select_final_top_k", return_value=5),
        patch("domain.services.rag_policy.compute_context_budget", return_value=1000),
    ):
        await rerank_and_enrich(
            "query",
            [],
            rag,
            ml,
            Breadth.NARROW,
            "legal",
            5,
            af,
            ctx,
            [],
            5,
            None,
            False,
            visibility_conditions=conditions,
            user_id=5,
            user_group_ids=[1, 2],
        )

    search.assert_awaited_once_with(
        "query",
        rag.retriever.fetch_k,
        af,
        rag,
        ml_clients=ml,
        visibility_conditions=conditions,
        user_id=5,
        user_group_ids=[1, 2],
    )


@pytest.fixture(autouse=True)
def mock_embedding_binding(monkeypatch):
    monkeypatch.setattr(
        "infrastructure.ml.rag.rag_retrieval.ensure_embedding_identity", lambda *args: "test:model:v1"
    )
