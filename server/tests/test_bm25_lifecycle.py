"""Regression tests for async load, invalidation and incremental BM25 updates."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from config import settings
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.bm25.snapshots import LocalBM25Snapshots
from infrastructure.storage.file_storage import LazyStorage
from infrastructure.utils.async_tasks import run_sync_io


@pytest.mark.asyncio
async def test_invalidation_reloads_immediately() -> None:
    loader = AsyncMock(side_effect=[BM25Index(["old"]), BM25Index(["fresh"])])
    registry = LocalBM25Snapshots(bm25_loader=loader)
    await registry.ensure_bm25_loaded()
    registry.invalidate_bm25()
    index = await registry.ensure_bm25_loaded()
    assert index is not None
    assert index.texts == ["fresh"]
    await registry.close()


@pytest.mark.asyncio
async def test_invalidation_discards_inflight_snapshot() -> None:
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def load() -> BM25Index:
        nonlocal calls
        calls += 1

        if calls == 1:
            started.set()
            await release.wait()
            return BM25Index(["stale"])

        return BM25Index(["fresh"])

    registry = LocalBM25Snapshots(bm25_loader=load)
    pending = asyncio.create_task(registry.ensure_bm25_loaded())
    await started.wait()
    registry.invalidate_bm25()
    release.set()
    index = await asyncio.wait_for(pending, 5)
    assert index is not None
    assert index.texts == ["fresh"]
    assert registry.bm25_index() is index
    await registry.close()


@pytest.mark.asyncio
async def test_failed_load_can_retry() -> None:
    loader = AsyncMock(side_effect=[OSError("storage unavailable"), BM25Index(["recovered"])])
    registry = LocalBM25Snapshots(bm25_loader=loader)
    with pytest.raises(OSError, match="storage unavailable"):
        await registry.ensure_bm25_loaded()
    index = await registry.ensure_bm25_loaded()
    assert index is not None
    assert index.texts == ["recovered"]
    await registry.close()


@pytest.mark.asyncio
async def test_config_rotation_discards_old_backend_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    buckets = []
    backends: list[SimpleNamespace] = []
    monkeypatch.setattr(settings, "s3_bucket", "old-bucket")

    def create(*_: object) -> SimpleNamespace:
        backend = SimpleNamespace(bucket=settings.s3_bucket, aclose=AsyncMock())
        backends.append(backend)
        return backend

    monkeypatch.setattr("infrastructure.storage.file_storage.create_storage", create)
    storage = LazyStorage(settings, settings)

    async def load() -> BM25Index:
        backend = storage.ensure()
        assert isinstance(backend, SimpleNamespace)
        buckets.append(backend.bucket)

        if len(buckets) == 1:
            started.set()
            await release.wait()

        return BM25Index([backend.bucket])

    registry = LocalBM25Snapshots(bm25_loader=load)
    pending = asyncio.create_task(registry.ensure_bm25_loaded())
    await started.wait()
    monkeypatch.setattr(settings, "s3_bucket", "new-bucket")
    storage.invalidate()
    registry.invalidate_bm25()
    release.set()
    index = await asyncio.wait_for(pending, 5)
    assert buckets == ["old-bucket", "new-bucket"]
    assert index is not None
    assert index.texts == ["new-bucket"]
    await registry.close()
    await storage.aclose()
    assert all(backend.aclose.await_count == 1 for backend in backends)


@pytest.mark.asyncio
async def test_registry_shutdown_drains_cancelled_snapshot_parse() -> None:
    started, release = threading.Event(), threading.Event()
    finished = Mock()

    def parse() -> BM25Index:
        started.set()

        if not release.wait(5):
            message = "parse not released"
            raise TimeoutError(message)

        finished()
        return BM25Index(["snapshot"])

    async def load() -> BM25Index:
        return await run_sync_io(parse)

    registry = LocalBM25Snapshots(bm25_loader=load)
    pending = asyncio.create_task(registry.ensure_bm25_loaded())
    closing = None
    try:
        assert await asyncio.to_thread(started.wait, 5)
        pending.cancel()
        closing = asyncio.create_task(registry.close())
        await asyncio.sleep(0)
        assert not pending.done()
        assert not closing.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, 5)
    await asyncio.wait_for(closing, 5)
    finished.assert_called_once()


@pytest.mark.asyncio
async def test_cancelled_registry_close_finishes_all_resource_cleanup() -> None:
    started, release = asyncio.Event(), asyncio.Event()
    events = []

    async def load() -> BM25Index:
        started.set()
        await release.wait()
        events.append("load")
        return BM25Index([])

    registry = LocalBM25Snapshots(bm25_loader=load)

    pending = asyncio.create_task(registry.ensure_bm25_loaded())
    await started.wait()
    closing = asyncio.create_task(registry.close())
    try:
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
    await pending
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, 5)
    assert events == ["load"]
    await registry.close()
    assert events == ["load"]


@pytest.mark.asyncio
async def test_cancelled_drain_waits_for_mutation_and_prevents_new_loads() -> None:
    started, release = threading.Event(), threading.Event()
    owner = LocalBM25Snapshots(AsyncMock(return_value=BM25Index(["before"])))
    await owner.ensure_bm25_loaded()

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
    await owner.close()
