"""Async snapshot orchestration keeps generation, mutation and cancellation safe."""

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from botocore.exceptions import ClientError

from infrastructure.bm25 import persistence
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.ml.clients.client_registry import MLClientRegistry


@pytest.mark.asyncio
async def test_ten_callers_download_and_parse_one_snapshot_off_loop(tmp_path, monkeypatch):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"texts": ["original"], "hashes": ["hash"]}))
    downloads = []
    parse_threads = []
    original = BM25Index.from_dict
    loop_thread = threading.get_ident()

    async def download(key):
        downloads.append(key)
        await asyncio.sleep(0)
        return path

    def parse(data):
        parse_threads.append(threading.get_ident())
        return original(data)

    monkeypatch.setattr(BM25Index, "from_dict", parse)
    storage = AsyncMock(download_to_temp=AsyncMock(side_effect=download))
    registry = MLClientRegistry(bm25_loader=lambda: persistence.load_bm25_index_from_s3(storage))
    snapshots = await asyncio.gather(*(registry.ensure_bm25_loaded() for _ in range(10)))
    assert all(index is snapshots[0] for index in snapshots)
    assert snapshots[0].hashes == ["hash"]
    assert downloads == [persistence.BM25_S3_KEY]
    assert len(parse_threads) == 1 and parse_threads[0] != loop_thread
    assert not path.exists()
    await registry.close()


@pytest.mark.asyncio
async def test_cached_access_and_update_never_load_storage():
    loader = AsyncMock(return_value=BM25Index(["snapshot"]))
    registry = MLClientRegistry(bm25_loader=loader)
    with pytest.raises(RuntimeError, match="not ready"):
        registry.bm25_index()
    with pytest.raises(RuntimeError, match="not ready"):
        with registry.bm25_update():
            pytest.fail("Unconfirmed empty index must not be created")
    loader.assert_not_awaited()
    index = await registry.ensure_bm25_loaded()
    assert registry.bm25_index() is index
    with registry.bm25_update() as updated:
        assert updated is index
    loader.assert_awaited_once()
    await registry.close()


@pytest.mark.asyncio
async def test_invalidation_during_async_download_discards_stale_snapshot():
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
    assert index.texts == ["fresh"] and calls == 2
    assert registry.bm25_index() is index
    await registry.close()


@pytest.mark.asyncio
async def test_cancelled_first_loader_does_not_poison_waiting_caller():
    started = asyncio.Event()
    calls = 0

    async def load():
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await asyncio.Event().wait()
        return BM25Index(["recovered"])

    registry = MLClientRegistry(bm25_loader=load)
    first = asyncio.create_task(registry.ensure_bm25_loaded())
    await started.wait()
    second = asyncio.create_task(registry.ensure_bm25_loaded())
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert (await asyncio.wait_for(second, 5)).texts == ["recovered"]
    assert calls == 2
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["NoSuchKey", "404", "NotFound", "AccessDenied", "NoSuchBucket"])
async def test_async_snapshot_missing_is_distinct_from_failed_load(code):
    failure = ClientError({"Error": {"Code": code}}, "GetObject")
    storage = AsyncMock(download_to_temp=AsyncMock(side_effect=failure))
    if code in {"NoSuchKey", "404", "NotFound"}:
        assert await persistence.load_bm25_index_from_s3(storage) is None
    else:
        with pytest.raises(persistence.BM25LoadError) as caught:
            await persistence.load_bm25_index_from_s3(storage)
        assert caught.value.__cause__ is failure


@pytest.mark.asyncio
async def test_async_snapshot_corruption_not_cached_as_missing(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text('{"texts":[null]}')
    storage = AsyncMock(download_to_temp=AsyncMock(return_value=path))
    with pytest.raises(persistence.BM25LoadError):
        await persistence.load_bm25_index_from_s3(storage)
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("parse_error", [ValueError, OSError])
async def test_cancelled_parse_failure_preserves_cancellation(tmp_path, monkeypatch, parse_error):
    path = tmp_path / "snapshot.json"
    path.write_text('{"texts":["snapshot"]}')
    started, release = threading.Event(), threading.Event()

    def parse(snapshot_path):
        started.set()
        if not release.wait(5):
            raise TimeoutError("test did not release parsing")
        raise parse_error("parse failed after cancellation")

    monkeypatch.setattr(persistence, "parse_bm25_snapshot", parse)
    storage = AsyncMock(download_to_temp=AsyncMock(return_value=path))
    registry = MLClientRegistry(bm25_loader=lambda: persistence.load_bm25_index_from_s3(storage))
    operation = registry.ensure_bm25_loaded()
    pending = asyncio.create_task(operation)
    try:
        assert await asyncio.to_thread(started.wait, 5)
        pending.cancel("request cancelled")
        release.set()
        with pytest.raises(asyncio.CancelledError, match="request cancelled"):
            await asyncio.wait_for(pending, 5)
        assert not path.exists()
        with pytest.raises(persistence.BM25SnapshotNotReady):
            registry.bm25_index()
    finally:
        release.set()
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_error", [PermissionError, OSError])
async def test_parse_cancellation_survives_cleanup_failure(tmp_path, monkeypatch, cleanup_error):
    path = tmp_path / "snapshot.json"
    path.write_text('{"texts":["snapshot"]}')
    started, release = threading.Event(), threading.Event()
    original_parse = persistence.parse_bm25_snapshot
    original_unlink = Path.unlink
    failure = cleanup_error("cannot remove snapshot")
    cleanup_attempted = threading.Event()

    def parse(snapshot_path):
        started.set()
        if not release.wait(5):
            raise TimeoutError("test did not release parsing")
        return original_parse(snapshot_path)

    def unlink(snapshot_path, *args, **kwargs):
        if snapshot_path == path:
            cleanup_attempted.set()
            raise failure
        return original_unlink(snapshot_path, *args, **kwargs)

    monkeypatch.setattr(persistence, "parse_bm25_snapshot", parse)
    monkeypatch.setattr(Path, "unlink", unlink)
    storage = AsyncMock(download_to_temp=AsyncMock(return_value=path))
    registry = MLClientRegistry(bm25_loader=lambda: persistence.load_bm25_index_from_s3(storage))
    operation = persistence.load_bm25_index_from_s3(storage)
    pending = asyncio.create_task(operation)
    try:
        assert await asyncio.to_thread(started.wait, 5)
        pending.cancel("request cancelled")
        release.set()
        with pytest.raises(asyncio.CancelledError, match="request cancelled"):
            await asyncio.wait_for(pending, 5)
        assert cleanup_attempted.is_set()
        with pytest.raises(persistence.BM25SnapshotNotReady):
            registry.bm25_index()
    finally:
        release.set()
        await registry.close()


@pytest.mark.asyncio
async def test_cleanup_failure_without_cancellation_remains_load_error(tmp_path, monkeypatch):
    path = tmp_path / "snapshot.json"
    path.write_text('{"texts":["snapshot"]}')
    failure = PermissionError("cannot remove snapshot")
    original_unlink = Path.unlink

    def unlink(snapshot_path, *args, **kwargs):
        if snapshot_path == path:
            raise failure
        return original_unlink(snapshot_path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    storage = AsyncMock(download_to_temp=AsyncMock(return_value=path))
    with pytest.raises(persistence.BM25LoadError) as caught:
        await persistence.load_bm25_index_from_s3(storage)
    assert caught.value.__cause__ is failure


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        b'{"texts":[42]}',
        b'{"texts":"not a list"}',
        b'{"texts":["text"],"doc_visibility":[{}]}',
        b'{"texts":["text"],"doc_owner_id":[]}',
        b'{"texts":["text"],"k1":"bad"}',
        b'bad json',
    ],
)
async def test_invalid_persisted_snapshot_is_retryable_load_failure(tmp_path, raw):
    path = tmp_path / "snapshot.json"
    path.write_bytes(raw)
    with pytest.raises(persistence.BM25LoadError) as caught:
        await persistence.load_bm25_index_from_s3(AsyncMock(download_to_temp=AsyncMock(return_value=path)))
    assert isinstance(caught.value.__cause__, ValueError)
    assert not path.exists()


@pytest.mark.asyncio
async def test_temp_directory_failure_does_not_become_cached_missing(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text('{"texts":["existing snapshot"],"hashes":["existing-hash"]}')
    failure = FileNotFoundError("temporary directory disappeared after GET")
    storage = AsyncMock(download_to_temp=AsyncMock(side_effect=[failure, path]))
    registry = MLClientRegistry(bm25_loader=lambda: persistence.load_bm25_index_from_s3(storage))
    try:
        with pytest.raises(persistence.BM25LoadError) as caught:
            await registry.ensure_bm25_loaded()
        assert caught.value.__cause__ is failure
        with pytest.raises(RuntimeError, match="not ready"):
            registry.bm25_index()
        assert (await registry.ensure_bm25_loaded()).hashes == ["existing-hash"]
    finally:
        await registry.close()
