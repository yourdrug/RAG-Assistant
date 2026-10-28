"""Corpus ownership, admission and shutdown without ML model lifecycle."""

from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from domain.shared.exceptions import ValidationError
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
    owner.install_corpus(old)
    async with owner.corpus_operation():
        key = _cache_key("question", "scope")
        owner.install_corpus(new)
        async with registry.corpus_operation():
            assert registry.qdrant_client() is old
            assert registry.corpus_snapshot == old.snapshot
            assert _cache_key("question", "scope") == key
        cast("Mock", old.client.close).assert_not_called()
    async with owner.corpus_operation():
        assert owner.qdrant_client() is new
        assert _cache_key("question", "scope") != key
    await registry.close()
    cast("Mock", old.client.close).assert_called_once()
    cast("Mock", new.client.close).assert_called_once()


@pytest.mark.asyncio
async def test_two_owners_restore_their_own_context() -> None:
    first, second = CorpusResources(Mock()), CorpusResources(Mock())
    old, new, other = corpus("old"), corpus("new"), corpus("other")
    first.install_corpus(old)
    second.install_corpus(other)
    async with first.corpus_operation():
        async with second.corpus_operation():
            assert second.qdrant_client() is other
        first.install_corpus(new)
        assert first.qdrant_client() is old
    await first.close()
    await second.close()


@pytest.mark.asyncio
async def test_readiness_denies_admission_but_nested_operation_keeps_lease() -> None:
    owner = CorpusResources(Mock())
    owner.install_corpus(corpus("old"))
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
async def test_cancelled_close_waits_for_lease_and_is_idempotent() -> None:
    owner = CorpusResources(Mock())
    client = corpus("old")
    owner.install_corpus(client)
    async with owner.corpus_operation():
        closing = asyncio.create_task(owner.close())
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        assert not closing.done()
        cast("Mock", client.client.close).assert_not_called()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, 5)
    await owner.close()
    cast("Mock", client.client.close).assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        owner.qdrant_client()


@pytest.mark.asyncio
async def test_retired_cleanup_failure_is_retried_at_shutdown() -> None:
    owner = CorpusResources(Mock())
    old, new = corpus("old"), corpus("new")
    cast("Mock", old.client.close).side_effect = [OSError("pool unavailable"), None]
    owner.install_corpus(old)
    owner.install_corpus(new)
    assert old.close_task is not None
    await old.close_task
    assert old in owner.retired_corpus_clients
    await owner.close()
    assert old.closed
    assert cast("Mock", old.client.close).call_count == 2
    cast("Mock", new.client.close).assert_called_once()
