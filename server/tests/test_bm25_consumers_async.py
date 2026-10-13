"""Consumers prepare sparse state asynchronously and preserve dense ACL fallback."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from domain.value_objects.chat_context import ChatContext
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility
from domain.services.access_control import get_visibility_conditions
from infrastructure.benchmark import answer_generators, retrieval, sweep_scoring
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.bm25.persistence import BM25LoadError
from infrastructure.metrics import metrics


@pytest.mark.asyncio
async def test_metrics_use_async_preparation_not_sync_accessor():
    clients = SimpleNamespace(
        ensure_bm25_loaded=AsyncMock(return_value=BM25Index(["text"], hashes=["hash"])),
        bm25_index=lambda: pytest.fail("Metrics must not trigger a synchronous load"),
    )
    await metrics.collect_bm25_metrics(clients)
    assert metrics.BM25_INDEX_SIZE._value.get() == 1


@pytest.mark.asyncio
async def test_metrics_without_registry_use_explicit_loader():
    await metrics.collect_bm25_metrics(None, bm25_loader=AsyncMock(return_value=BM25Index(["a", "b"])))
    assert metrics.BM25_INDEX_SIZE._value.get() == 2


def test_sparse_cpu_search_uses_explicit_snapshot_only():
    clients = SimpleNamespace(bm25_index=lambda: pytest.fail("CPU path must not load S3"))
    index = BM25Index(
        ["benchmark evidence"], hashes=["hash"], doc_visibility=[DocumentVisibility.INTERNAL_PUBLIC.value]
    )
    conditions = get_visibility_conditions(UserKind.INTERNAL, 1, [], for_list=False, user_role=UserRole.ADMIN)
    assert (
        retrieval.search_sparse(
            "evidence",
            10,
            clients,
            bm25_index=index,
            visibility_conditions=conditions,
            user_id=1,
            user_group_ids=[],
        )[0][0]
        == "hash"
    )
    assert retrieval.search_sparse("evidence", 10, clients, bm25_index=None) == []


@pytest.mark.asyncio
async def test_standalone_retries_failed_preparation_and_passes_snapshot(monkeypatch):
    ctx = ChatContext(user_id=0, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN)
    generator = answer_generators.StandaloneBenchmarkGenerator(4, 10, ctx)
    index = BM25Index(["text"])
    generator._clients.ensure_bm25_loaded = AsyncMock(side_effect=[BM25LoadError("offline"), index])
    llm = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(retrieval, "build_llm", lambda *a, **k: llm)
    snapshots, filters = [], []

    def retrieve(*args, **kwargs):
        snapshots.append(kwargs["bm25_index"])
        filters.append(kwargs["access_filter"])
        return []

    monkeypatch.setattr(retrieval, "retrieve_with_scores_hybrid", retrieve)
    monkeypatch.setattr(answer_generators.judge, "get_rag_answer_with_usage", lambda *a: ("answer", None))
    try:
        await generator.generate({"question": "q"}, ctx)
        await generator.generate({"question": "q"}, ctx)
    finally:
        await generator.close()
    assert snapshots == [None, index]
    assert all(af is generator._access_filter for af in filters)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("bug"), asyncio.CancelledError()])
async def test_snapshot_preparation_does_not_hide_programming_errors_or_cancellation(error):
    with pytest.raises(type(error)):
        await sweep_scoring.load_candidate_snapshot(None, bm25_loader=AsyncMock(side_effect=error))


@pytest.mark.asyncio
async def test_sweep_failed_load_then_recovery_is_not_cached():
    index = BM25Index(["text"])
    loader = AsyncMock(side_effect=[BM25LoadError("offline"), index])
    assert await sweep_scoring.load_candidate_snapshot(None, bm25_loader=loader) is None
    assert await sweep_scoring.load_candidate_snapshot(None, bm25_loader=loader) is index
