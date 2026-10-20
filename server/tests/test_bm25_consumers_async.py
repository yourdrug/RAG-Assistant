"""Benchmark consumers use Qdrant sparse payloads without S3 snapshot loading."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.rag.context import RagContext
from domain.identity.roles import UserKind, UserRole
from domain.knowledge.access_control import get_visibility_conditions
from infrastructure.benchmark import answer_generators, retrieval
from infrastructure.repositories.vector.acl import build_qdrant_filter


def test_benchmark_sparse_passes_scope_and_collects_sparse_only_payload():
    scope = build_qdrant_filter({"id": 1, "kind": "internal", "role": "admin"}, [])
    client = MagicMock()
    client.search.return_value = [
        SimpleNamespace(
            id=17,
            score=2.0,
            payload={"page_content": "evidence", "metadata": {"chunk_id": 17}},
        )
    ]
    clients = SimpleNamespace(qdrant_client=lambda: client)
    documents = {}
    result = retrieval.search_sparse(
        "evidence",
        10,
        clients,
        access_filter=scope,
        documents=documents,
        visibility_conditions=get_visibility_conditions("internal", 1, [], for_list=False),
        user_id=1,
        user_group_ids=[],
    )
    assert result == [("17", 2.0)]
    assert documents["17"][1].page_content == "evidence"
    assert client.search.call_args.kwargs["query_filter"] is scope


@pytest.mark.asyncio
async def test_standalone_uses_qdrant_without_loading_snapshot(monkeypatch):
    ctx = RagContext(user_id=0, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN)
    generator = answer_generators.StandaloneBenchmarkGenerator(4, 10, ctx)
    loader = AsyncMock(side_effect=AssertionError("obsolete snapshot load"))
    generator._clients.ensure_bm25_loaded = loader
    llm = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(retrieval, "build_llm", lambda *a, **k: llm)
    filters = []

    def retrieve(*args, **kwargs):
        assert "bm25_index" not in kwargs
        filters.append(kwargs["access_filter"])
        return []

    monkeypatch.setattr(retrieval, "retrieve_with_scores_hybrid", retrieve)
    monkeypatch.setattr(answer_generators.judge, "get_rag_answer_with_usage", lambda *a: ("answer", None))
    try:
        await generator.generate({"question": "q"}, ctx)
    finally:
        await generator.close()
    loader.assert_not_awaited()
    assert filters == [generator._access_filter]
