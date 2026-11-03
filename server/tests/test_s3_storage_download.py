"""Streaming content, bounded buffers and cancellation cleanup."""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import threading
from typing import Any

import pytest

from infrastructure.bm25 import persistence
from infrastructure.bm25.snapshots import LocalBM25Snapshots
from infrastructure.storage import file_storage
from tests.s3_fakes import Body
from tests.test_s3_storage_async import client_harness


def track_temp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Path]:
    original = tempfile.NamedTemporaryFile
    paths = []

    def create(**kwargs: Any) -> tempfile._TemporaryFileWrapper[bytes]:
        assert kwargs.pop("mode", "w+b") == "w+b"
        result = original(mode="w+b", dir=tmp_path, **kwargs)
        paths.append(Path(result.name))
        return result

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", create)
    return paths


@pytest.mark.asyncio
@pytest.mark.parametrize("blocks", [[], [b"a" * (1024 * 1024), b"remaining"]])
async def test_download_content_and_closed_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, blocks: list[bytes]
) -> None:
    storage, client, _, _ = client_harness(monkeypatch)
    body = Body(blocks)
    client.get_object.return_value = {"Body": body}
    paths = track_temp(monkeypatch, tmp_path)
    path = await storage.download_to_temp("report.pdf")
    assert path == paths[0]
    assert path.suffix == ".pdf"
    assert path.read_bytes() == b"".join(blocks)
    assert body.closed
    assert set(body.read_sizes) == {1024 * 1024}
    path.unlink()
    await storage.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["get", "read", "write", "cancel_read"])
async def test_failed_download_closes_body_and_removes_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    storage, client, _, _ = client_harness(monkeypatch)
    body = Body([b"first"], error=OSError("read failed") if failure == "read" else None)
    client.get_object.return_value = {"Body": body}
    paths = track_temp(monkeypatch, tmp_path)

    if failure == "get":
        client.get_object.side_effect = OSError("get failed")

    if failure == "write":

        def fail_write(*args: Any) -> None:
            msg = "write failed"
            raise OSError(msg)

        monkeypatch.setattr(file_storage, "write_download_block", fail_write)

    if failure == "cancel_read":
        body.release = asyncio.Event()
        pending = asyncio.create_task(storage.download_to_temp("report.pdf"))
        await body.started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        with pytest.raises(OSError, match="failed"):
            await storage.download_to_temp("report.pdf")

    assert not any(path.exists() for path in paths)

    if failure != "get":
        assert body.closed

    await storage.aclose()


@pytest.mark.asyncio
async def test_cancel_thread_write_waits_before_closing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    storage, client, _, _ = client_harness(monkeypatch)
    body = Body([b"content"])
    client.get_object.return_value = {"Body": body}
    paths = track_temp(monkeypatch, tmp_path)
    started, release = threading.Event(), threading.Event()
    closed_during_write = []

    def delayed_write(handle: tempfile._TemporaryFileWrapper[bytes], block: bytes) -> None:
        started.set()

        if not release.wait(5):
            msg = "writer not released"
            raise TimeoutError(msg)

        closed_during_write.append(handle.closed)
        handle.write(block)

    monkeypatch.setattr(file_storage, "write_download_block", delayed_write)
    pending = asyncio.create_task(storage.download_to_temp("report.pdf"))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done()
        assert paths[0].exists()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, 5)
    assert closed_during_write == [False]
    assert body.closed
    assert not paths[0].exists()
    await storage.aclose()


@pytest.mark.asyncio
async def test_get_holds_semaphore_until_body_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    storage, client, _, _ = client_harness(monkeypatch, max_concurrent=1)
    body = Body([])
    body.release = asyncio.Event()
    client.get_object.return_value = {"Body": body}
    pending = asyncio.create_task(storage.download_to_temp("a.txt"))
    await body.started.wait()
    upload = asyncio.create_task(storage.upload_file("b.txt", b"x"))
    await asyncio.sleep(0)
    client.put_object.assert_not_awaited()
    body.release.set()
    path, _ = await asyncio.gather(pending, upload)
    assert body.closed
    path.unlink()
    await storage.aclose()


@pytest.mark.asyncio
async def test_cancel_during_successful_close_removes_unreturned_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    storage, client, _, _ = client_harness(monkeypatch)
    body = Body([b"content"])
    client.get_object.return_value = {"Body": body}
    paths = track_temp(monkeypatch, tmp_path)
    original = file_storage.close_download
    started, release = threading.Event(), threading.Event()

    def delayed_close(handle: tempfile._TemporaryFileWrapper[bytes], *, remove: bool) -> None:
        if not remove:
            started.set()

            if not release.wait(5):
                msg = "close not released"
                raise TimeoutError(msg)

        original(handle, remove=remove)

    monkeypatch.setattr(file_storage, "close_download", delayed_close)
    pending = asyncio.create_task(storage.download_to_temp("a.txt"))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        pending.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, 5)
    assert body.closed
    assert not paths[0].exists()
    await storage.aclose()


class CleanupFailure:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, directory: Path, phase: str, failure: str) -> None:
        self.monkeypatch, self.directory, self.phase, self.failure = monkeypatch, directory, phase, failure
        self.original_create = tempfile.NamedTemporaryFile
        self.original_unlink = Path.unlink
        self.started, self.release = threading.Event(), threading.Event()
        self.cleanup_attempted = threading.Event()
        self.handles: list[tempfile._TemporaryFileWrapper[bytes]] = []
        self.install()

    def create(self, **kwargs: Any) -> tempfile._TemporaryFileWrapper[bytes]:
        assert kwargs.pop("mode", "w+b") == "w+b"
        handle = self.original_create(mode="w+b", dir=self.directory, **kwargs)
        self.handles.append(handle)
        original_close = handle.close

        def close() -> None:
            original_close()

            if self.failure == "close":
                self.cleanup_attempted.set()
                msg = "cannot close download"
                raise PermissionError(msg)

        self.monkeypatch.setattr(handle, "close", close)
        self.started.set()

        if self.phase == "creation" and not self.release.wait(5):
            msg = "creation not released"
            raise TimeoutError(msg)

        return handle

    def unlink(self, path: Path, *args: Any, **kwargs: Any) -> None:
        if path.parent == self.directory and self.failure == "unlink":
            self.cleanup_attempted.set()
            msg = "cannot remove download"
            raise PermissionError(msg)

        self.original_unlink(path, *args, **kwargs)

    def remove_files(self) -> None:
        for handle in self.handles:
            self.original_unlink(Path(handle.name), missing_ok=True)

    def install(self) -> None:
        def unlink(path: Path, *args: Any, **kwargs: Any) -> None:
            self.unlink(path, *args, **kwargs)

        self.monkeypatch.setattr(tempfile, "NamedTemporaryFile", self.create)
        self.monkeypatch.setattr(Path, "unlink", unlink)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["read", "creation"])
@pytest.mark.parametrize("failure", ["close", "unlink"])
@pytest.mark.parametrize("consumer", ["storage", "loader"])
async def test_download_cancellation_survives_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase: str, failure: str, consumer: str
) -> None:
    storage, client, _, _ = client_harness(monkeypatch)
    body = Body([])
    body.release = asyncio.Event()
    client.get_object.return_value = {"Body": body}
    faults = CleanupFailure(monkeypatch, tmp_path, phase, failure)
    registry = LocalBM25Snapshots(bm25_loader=lambda: persistence.load_bm25_index_from_s3(storage))
    operation = {
        "storage": lambda: storage.download_to_temp("report.pdf"),
        "loader": registry.ensure_bm25_loaded,
    }[consumer]()
    pending = asyncio.create_task(operation)
    try:
        if phase == "creation":
            assert await asyncio.to_thread(faults.started.wait, 5)
        else:
            await asyncio.wait_for(body.started.wait(), 5)

        pending.cancel("request cancelled")
        faults.release.set()
        with pytest.raises(asyncio.CancelledError, match="request cancelled"):
            await asyncio.wait_for(pending, 5)
        assert pending.cancelled()
        assert faults.cleanup_attempted.is_set()
        assert faults.handles[0].closed
        assert body.closed
        assert storage._active == 0
        with pytest.raises(persistence.BM25SnapshotNotReadyError):
            registry.bm25_index()
    finally:
        faults.release.set()

        faults.remove_files()

        await registry.close()
        await storage.aclose()


@pytest.mark.asyncio
async def test_cancelled_temp_creation_failure_preserves_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, client, _, _ = client_harness(monkeypatch)
    body = Body([])
    client.get_object.return_value = {"Body": body}
    started, release = threading.Event(), threading.Event()

    def create(**kwargs: Any) -> tempfile._TemporaryFileWrapper[bytes]:
        started.set()

        if not release.wait(5):
            msg = "creation not released"
            raise TimeoutError(msg)

        msg = "cannot create download"
        raise PermissionError(msg)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", create)
    pending = asyncio.create_task(storage.download_to_temp("report.pdf"))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        pending.cancel("request cancelled")
        release.set()
        with pytest.raises(asyncio.CancelledError, match="request cancelled"):
            await asyncio.wait_for(pending, 5)
        assert body.closed
        assert storage._active == 0
    finally:
        release.set()
        await storage.aclose()
