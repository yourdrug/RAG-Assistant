"""Corpus ownership, admission and shutdown without ML model lifecycle."""

import asyncio
import threading
from unittest.mock import AsyncMock, Mock

import pytest

from domain.shared.exceptions import ValidationError
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.corpus.resources import CorpusResources
from infrastructure.corpus.snapshot import BoundQdrantClient, CorpusSnapshot
from infrastructure.ml.cache.answers import _cache_key
from infrastructure.ml.clients.client_registry import MLClientRegistry


def corpus(generation: str) -> BoundQdrantClient:
    return BoundQdrantClient(
        Mock(), CorpusSnapshot("http://qdrant:6333", "docs", generation, "encoder", generation)
    )


@pytest.mark.asyncio
async def test_nested_owner_and_registry_operations_keep_pair_and_namespace() -> None:
    owner = CorpusResources(Mock())
    registry = MLClientRegistry(corpus_resources=owner)
    old, new = corpus("old"), corpus("new")
    old_index, new_index = BM25Index(["old"]), BM25Index(["new"])
    owner.install_corpus(old, old_index)
    async with owner.corpus_operation():
        key = _cache_key("question", "scope")
        owner.install_corpus(new, new_index)
        async with registry.corpus_operation():
            assert registry.qdrant_client() is old
            assert registry.corpus_snapshot == old.snapshot
            assert registry.bm25_index() is old_index
            assert await registry.ensure_bm25_loaded() is old_index
            assert _cache_key("question", "scope") == key
        old.client.close.assert_not_called()
    async with owner.corpus_operation():
        assert owner.qdrant_client() is new
        assert await owner.ensure_bm25_loaded() is new_index
        assert _cache_key("question", "scope") != key
    await registry.close()
    old.client.close.assert_called_once()
    new.client.close.assert_called_once()


@pytest.mark.asyncio
async def test_two_owners_restore_their_own_context() -> None:
    first, second = CorpusResources(Mock()), CorpusResources(Mock())
    old, new, other = corpus("old"), corpus("new"), corpus("other")
    first.install_corpus(old, None)
    second.install_corpus(other, None)
    async with first.corpus_operation():
        async with second.corpus_operation():
            assert second.qdrant_client() is other
        first.install_corpus(new, None)
        assert first.qdrant_client() is old
    await first.close()
    await second.close()


@pytest.mark.asyncio
async def test_readiness_denies_admission_but_nested_operation_keeps_lease() -> None:
    owner = CorpusResources(Mock())
    owner.install_corpus(corpus("old"), None)
    check = AsyncMock(return_value=False)
    owner.bind_readiness(check)
    with pytest.raises(ValidationError, match="not ready"):
        async with owner.corpus_operation():
            pytest.fail("admitted an unready corpus")
    check.return_value = True
    async with owner.corpus_operation():
        check.return_value = False
        async with owner.corpus_operation():
            snapshot = owner.corpus_snapshot
            assert snapshot is not None
            assert snapshot.generation == "old"
    assert check.await_count == 2
    await owner.close()


@pytest.mark.asyncio
async def test_install_discards_inflight_legacy_load() -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def load() -> BM25Index:
        started.set()
        await release.wait()
        return BM25Index(["stale"])

    owner = CorpusResources(Mock(), bm25_loader=load)
    pending = asyncio.create_task(owner.ensure_bm25_loaded())
    await started.wait()
    installed = BM25Index(["installed"])
    owner.install_corpus(corpus("new"), installed)
    release.set()
    assert await asyncio.wait_for(pending, 5) is installed
    assert owner.bm25_index() is installed
    await owner.close()


@pytest.mark.asyncio
async def test_cancelled_close_waits_for_lease_and_is_idempotent() -> None:
    owner = CorpusResources(Mock())
    client = corpus("old")
    owner.install_corpus(client, None)
    async with owner.corpus_operation():
        closing = asyncio.create_task(owner.close())
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        assert not closing.done()
        client.client.close.assert_not_called()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, 5)
    await owner.close()
    client.client.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        await owner.ensure_bm25_loaded()


@pytest.mark.asyncio
async def test_retired_cleanup_failure_is_retried_at_shutdown() -> None:
    owner = CorpusResources(Mock())
    old, new = corpus("old"), corpus("new")
    old.client.close.side_effect = [OSError("pool unavailable"), None]
    owner.install_corpus(old, None)
    owner.install_corpus(new, None)
    assert old.close_task is not None
    await old.close_task
    assert old in owner.retired_corpus_clients
    await owner.close()
    assert old.closed
    assert old.client.close.call_count == 2
    new.client.close.assert_called_once()


@pytest.mark.asyncio
async def test_cancelled_drain_waits_for_mutation_and_prevents_new_loads() -> None:
    started, release = threading.Event(), threading.Event()
    owner = CorpusResources(Mock(), bm25_loader=AsyncMock())
    owner.install_corpus(corpus("old"), BM25Index(["before"]))

    def mutate() -> None:
        with owner.bm25_update() as index:
            started.set()

            if not release.wait(5):
                message = "mutation not released"
                raise TimeoutError(message)

            index.add_text("after", "hash")

    mutation = asyncio.create_task(asyncio.to_thread(mutate))
    draining = None
    try:
        assert await asyncio.to_thread(started.wait, 5)
        draining = asyncio.create_task(owner.drain())
        await asyncio.sleep(0)
        draining.cancel()
        await asyncio.sleep(0)
        assert not draining.done()
        with pytest.raises(RuntimeError, match="closed"):
            await owner.ensure_bm25_loaded()
    finally:
        release.set()
        await asyncio.wait_for(mutation, 5)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(draining, 5)
    await owner.drain()
    await owner.close()
