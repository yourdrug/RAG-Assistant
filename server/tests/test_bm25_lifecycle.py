"""Regression tests for async load, invalidation and incremental BM25 updates."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from config import settings
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.storage.file_storage import LazyStorage, run_storage_io


@pytest.mark.asyncio
async def test_invalidation_reloads_immediately():
    loader = AsyncMock(side_effect=[BM25Index(["old"]), BM25Index(["fresh"])])
    registry = MLClientRegistry(bm25_loader=loader)
    await registry.ensure_bm25_loaded()
    registry.invalidate_bm25()
    assert (await registry.ensure_bm25_loaded()).texts == ["fresh"]
    await registry.close()


@pytest.mark.asyncio
async def test_invalidation_discards_inflight_snapshot():
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def load():
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
            return BM25Index(["stale"])
        return BM25Index(["fresh"])

    registry = MLClientRegistry(bm25_loader=load)
    pending = asyncio.create_task(registry.ensure_bm25_loaded())
    await started.wait()
    registry.invalidate_bm25()
    release.set()
    index = await asyncio.wait_for(pending, 5)
    assert index.texts == ["fresh"]
    assert registry.bm25_index() is index
    await registry.close()


@pytest.mark.asyncio
async def test_failed_load_can_retry():
    loader = AsyncMock(side_effect=[OSError("storage unavailable"), BM25Index(["recovered"])])
    registry = MLClientRegistry(bm25_loader=loader)
    with pytest.raises(OSError, match="storage unavailable"):
        await registry.ensure_bm25_loaded()
    assert (await registry.ensure_bm25_loaded()).texts == ["recovered"]
    await registry.close()


@pytest.mark.asyncio
async def test_config_rotation_discards_old_backend_snapshot(monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    buckets = []
    backends = []
    monkeypatch.setattr(settings, "s3_bucket", "old-bucket")

    def create():
        backend = SimpleNamespace(bucket=settings.s3_bucket, aclose=AsyncMock())
        backends.append(backend)
        return backend

    monkeypatch.setattr("infrastructure.storage.file_storage.create_storage", create)
    storage = LazyStorage()

    async def load():
        backend = storage.ensure()
        buckets.append(backend.bucket)
        if len(buckets) == 1:
            started.set()
            await release.wait()
        return BM25Index([backend.bucket])

    registry = MLClientRegistry(bm25_loader=load)
    pending = asyncio.create_task(registry.ensure_bm25_loaded())
    await started.wait()
    monkeypatch.setattr(settings, "s3_bucket", "new-bucket")
    storage.invalidate()
    registry.invalidate_bm25()
    release.set()
    index = await asyncio.wait_for(pending, 5)
    assert buckets == ["old-bucket", "new-bucket"]
    assert index.texts == ["new-bucket"]
    await registry.close()
    await storage.aclose()
    assert all(backend.aclose.await_count == 1 for backend in backends)


@pytest.mark.asyncio
async def test_registry_shutdown_drains_cancelled_snapshot_parse():
    started, release = threading.Event(), threading.Event()
    finished = Mock()

    def parse():
        started.set()
        if not release.wait(5):
            raise TimeoutError("parse not released")
        finished()
        return BM25Index(["snapshot"])

    async def load():
        return await run_storage_io(parse)

    registry = MLClientRegistry(bm25_loader=load)
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
async def test_cancelled_registry_close_finishes_all_resource_cleanup():
    started, release = asyncio.Event(), asyncio.Event()
    events = []

    async def load():
        started.set()
        await release.wait()
        events.append("load")
        return BM25Index([])

    registry = MLClientRegistry(bm25_loader=load)

    async def close_ml():
        events.append("ml")

    registry._embeddings = SimpleNamespace(close=AsyncMock(side_effect=close_ml))
    registry._qdrant_client = SimpleNamespace(close=lambda: events.append("qdrant"))
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
    assert events == ["load", "ml", "qdrant"]
    await registry.close()
    assert events == ["load", "ml", "qdrant"]
