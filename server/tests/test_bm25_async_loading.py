"""Async snapshot orchestration keeps generation, mutation and cancellation safe."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
from typing import TYPE_CHECKING, NoReturn
from unittest.mock import AsyncMock

from botocore.exceptions import ClientError
import pytest

from infrastructure.bm25 import persistence
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.bm25.snapshots import LocalBM25Snapshots


if TYPE_CHECKING:
    from infrastructure.bm25.bm25_index import BM25Snapshot


@pytest.mark.asyncio
async def test_ten_callers_download_and_parse_one_snapshot_off_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"texts": ["original"], "hashes": ["hash"]}))
    downloads = []
    parse_threads = []
    original = BM25Index.from_dict
    loop_thread = threading.get_ident()

    async def download(key: str) -> Path:
        downloads.append(key)
        await asyncio.sleep(0)
        return path

    def parse(data: BM25Snapshot) -> BM25Index:
        parse_threads.append(threading.get_ident())
        return original(data)

    monkeypatch.setattr(BM25Index, "from_dict", parse)
    storage = AsyncMock(download_to_temp=AsyncMock(side_effect=download))
    registry = LocalBM25Snapshots(bm25_loader=lambda: persistence.load_bm25_index_from_s3(storage))
    snapshots = await asyncio.gather(*(registry.ensure_bm25_loaded() for _ in range(10)))
    assert all(index is snapshots[0] for index in snapshots)
    assert snapshots[0] is not None
    assert snapshots[0].hashes == ["hash"]
    assert downloads == [persistence.BM25_S3_KEY]
    assert len(parse_threads) == 1
    assert parse_threads[0] != loop_thread
    assert not path.exists()
    await registry.close()


@pytest.mark.asyncio
async def test_cached_access_and_update_never_load_storage() -> None:
    loader = AsyncMock(return_value=BM25Index(["snapshot"]))
    registry = LocalBM25Snapshots(bm25_loader=loader)
    with pytest.raises(RuntimeError, match="not ready"):
        registry.bm25_index()
    with pytest.raises(RuntimeError, match="not ready"), registry.bm25_update():
        pytest.fail("Unconfirmed empty index must not be created")
    loader.assert_not_awaited()
    index = await registry.ensure_bm25_loaded()
    assert registry.bm25_index() is index
    with registry.bm25_update() as updated:
        assert updated is index
    loader.assert_awaited_once()
    await registry.close()


@pytest.mark.asyncio
async def test_invalidation_during_async_download_discards_stale_snapshot() -> None:
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
    assert calls == 2
    assert registry.bm25_index() is index
    await registry.close()


@pytest.mark.asyncio
async def test_cancelled_first_loader_does_not_poison_waiting_caller() -> None:
    started = asyncio.Event()
    calls = 0

    async def load() -> BM25Index:
        nonlocal calls
        calls += 1

        if calls == 1:
            started.set()
            await asyncio.Event().wait()

        return BM25Index(["recovered"])

    registry = LocalBM25Snapshots(bm25_loader=load)
    first = asyncio.create_task(registry.ensure_bm25_loaded())
    await started.wait()
    second = asyncio.create_task(registry.ensure_bm25_loaded())
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    index = await asyncio.wait_for(second, 5)
    assert index is not None
    assert index.texts == ["recovered"]
    assert calls == 2
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["NoSuchKey", "404", "NotFound", "AccessDenied", "NoSuchBucket"])
async def test_async_snapshot_missing_is_distinct_from_failed_load(code: str) -> None:
    failure = ClientError({"Error": {"Code": code}}, "GetObject")
    storage = AsyncMock(download_to_temp=AsyncMock(side_effect=failure))

    if code in {"NoSuchKey", "404", "NotFound"}:
        assert await persistence.load_bm25_index_from_s3(storage) is None
    else:
        with pytest.raises(persistence.BM25LoadError) as caught:
            await persistence.load_bm25_index_from_s3(storage)
        assert caught.value.__cause__ is failure


@pytest.mark.asyncio
async def test_async_snapshot_corruption_not_cached_as_missing(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text('{"texts":[null]}')
    storage = AsyncMock(download_to_temp=AsyncMock(return_value=path))
    with pytest.raises(persistence.BM25LoadError):
        await persistence.load_bm25_index_from_s3(storage)
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("parse_error", [ValueError, OSError])
async def test_cancelled_parse_failure_preserves_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parse_error: type[Exception]
) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text('{"texts":["snapshot"]}')
    started, release = threading.Event(), threading.Event()

    def parse(snapshot_path: Path) -> NoReturn:
        started.set()

        if not release.wait(5):
            message = "test did not release parsing"
            raise TimeoutError(message)

        message = "parse failed after cancellation"
        raise parse_error(message)

    monkeypatch.setattr(persistence, "parse_bm25_snapshot", parse)
    storage = AsyncMock(download_to_temp=AsyncMock(return_value=path))
    registry = LocalBM25Snapshots(bm25_loader=lambda: persistence.load_bm25_index_from_s3(storage))
    operation = registry.ensure_bm25_loaded()
    pending = asyncio.create_task(operation)
    try:
        assert await asyncio.to_thread(started.wait, 5)
        pending.cancel("request cancelled")
        release.set()
        with pytest.raises(asyncio.CancelledError, match="request cancelled"):
            await asyncio.wait_for(pending, 5)
        assert not path.exists()
        with pytest.raises(persistence.BM25SnapshotNotReadyError):
            registry.bm25_index()
    finally:
        release.set()
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_error", [PermissionError, OSError])
async def test_parse_cancellation_survives_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_error: type[OSError]
) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text('{"texts":["snapshot"]}')
    started, release = threading.Event(), threading.Event()
    original_parse = persistence.parse_bm25_snapshot
    failure = cleanup_error("cannot remove snapshot")
    cleanup_attempted = threading.Event()

    def parse(snapshot_path: Path) -> BM25Index:
        started.set()

        if not release.wait(5):
            message = "test did not release parsing"
            raise TimeoutError(message)

        return original_parse(snapshot_path)

    patch_unlink_failure(monkeypatch, path, failure, cleanup_attempted)

    monkeypatch.setattr(persistence, "parse_bm25_snapshot", parse)
    storage = AsyncMock(download_to_temp=AsyncMock(return_value=path))
    registry = LocalBM25Snapshots(bm25_loader=lambda: persistence.load_bm25_index_from_s3(storage))
    operation = persistence.load_bm25_index_from_s3(storage)
    pending = asyncio.create_task(operation)
    try:
        assert await asyncio.to_thread(started.wait, 5)
        pending.cancel("request cancelled")
        release.set()
        with pytest.raises(asyncio.CancelledError, match="request cancelled"):
            await asyncio.wait_for(pending, 5)
        assert cleanup_attempted.is_set()
        with pytest.raises(persistence.BM25SnapshotNotReadyError):
            registry.bm25_index()
    finally:
        release.set()
        await registry.close()


@pytest.mark.asyncio
async def test_cleanup_failure_without_cancellation_remains_load_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text('{"texts":["snapshot"]}')
    failure = PermissionError("cannot remove snapshot")
    patch_unlink_failure(monkeypatch, path, failure)
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
        b"bad json",
    ],
)
async def test_invalid_persisted_snapshot_is_retryable_load_failure(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "snapshot.json"
    path.write_bytes(raw)
    with pytest.raises(persistence.BM25LoadError) as caught:
        await persistence.load_bm25_index_from_s3(AsyncMock(download_to_temp=AsyncMock(return_value=path)))
    assert isinstance(caught.value.__cause__, ValueError)
    assert not path.exists()


@pytest.mark.asyncio
async def test_temp_directory_failure_does_not_become_cached_missing(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text('{"texts":["existing snapshot"],"hashes":["existing-hash"]}')
    failure = FileNotFoundError("temporary directory disappeared after GET")
    storage = AsyncMock(download_to_temp=AsyncMock(side_effect=[failure, path]))
    registry = LocalBM25Snapshots(bm25_loader=lambda: persistence.load_bm25_index_from_s3(storage))
    try:
        with pytest.raises(persistence.BM25LoadError) as caught:
            await registry.ensure_bm25_loaded()
        assert caught.value.__cause__ is failure
        with pytest.raises(RuntimeError, match="not ready"):
            registry.bm25_index()
        index = await registry.ensure_bm25_loaded()
        assert index is not None
        assert index.hashes == ["existing-hash"]
    finally:
        await registry.close()


def patch_unlink_failure(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    failure: OSError,
    attempted: threading.Event | None = None,
) -> None:
    original_unlink = Path.unlink

    def unlink(snapshot_path: Path, *, missing_ok: bool = False) -> None:
        if snapshot_path == path:
            if attempted is not None:
                attempted.set()

            raise failure

        original_unlink(snapshot_path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
