"""Streaming content, bounded buffers and cancellation cleanup."""

import asyncio
import threading
from pathlib import Path

import pytest

from test_s3_storage_async import client_harness
from infrastructure.storage import file_storage
from infrastructure.bm25 import persistence
from infrastructure.bm25.bm25_updater import BM25IndexAdapter
from infrastructure.ml.clients.client_registry import MLClientRegistry


class Body:
    def __init__(self, blocks, error=None):
        self.blocks = iter(blocks)
        self.error = error
        self.closed = False
        self.read_sizes = []
        self.started = asyncio.Event()
        self.release = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def read(self, size):
        self.read_sizes.append(size)
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        block = next(self.blocks, None)
        if block is not None:
            return block
        if self.error:
            raise self.error
        return b""


def track_temp(monkeypatch, tmp_path):
    original = file_storage.tempfile.NamedTemporaryFile
    paths = []

    def create(**kwargs):
        result = original(dir=tmp_path, **kwargs)
        paths.append(Path(result.name))
        return result

    monkeypatch.setattr(file_storage.tempfile, "NamedTemporaryFile", create)
    return paths


@pytest.mark.asyncio
@pytest.mark.parametrize("blocks", [[], [b"a" * (1024 * 1024), b"remaining"]])
async def test_download_content_and_closed_body(monkeypatch, tmp_path, blocks):
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
async def test_failed_download_closes_body_and_removes_partial_file(monkeypatch, tmp_path, failure):
    storage, client, _, _ = client_harness(monkeypatch)
    body = Body([b"first"], error=OSError("read failed") if failure == "read" else None)
    client.get_object.return_value = {"Body": body}
    paths = track_temp(monkeypatch, tmp_path)
    if failure == "get":
        client.get_object.side_effect = OSError("get failed")
    if failure == "write":

        def fail_write(*args):
            raise OSError("write failed")

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
async def test_cancel_thread_write_waits_before_closing_file(monkeypatch, tmp_path):
    storage, client, _, _ = client_harness(monkeypatch)
    body = Body([b"content"])
    client.get_object.return_value = {"Body": body}
    paths = track_temp(monkeypatch, tmp_path)
    started, release = threading.Event(), threading.Event()
    closed_during_write = []

    def delayed_write(handle, block):
        started.set()
        if not release.wait(5):
            raise TimeoutError("writer not released")
        closed_during_write.append(handle.closed)
        return handle.write(block)

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
async def test_get_holds_semaphore_until_body_closed(monkeypatch):
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
async def test_cancel_during_successful_close_removes_unreturned_file(monkeypatch, tmp_path):
    storage, client, _, _ = client_harness(monkeypatch)
    body = Body([b"content"])
    client.get_object.return_value = {"Body": body}
    paths = track_temp(monkeypatch, tmp_path)
    original = file_storage.close_download
    started, release = threading.Event(), threading.Event()

    def delayed_close(handle, *, remove):
        if not remove:
            started.set()
            if not release.wait(5):
                raise TimeoutError("close not released")
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


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["read", "creation"])
@pytest.mark.parametrize("failure", ["close", "unlink"])
@pytest.mark.parametrize("consumer", ["storage", "loader", "mutation"])
async def test_download_cancellation_survives_cleanup_failure(
    monkeypatch, tmp_path, phase, failure, consumer
):
    storage, client, _, _ = client_harness(monkeypatch)
    body = Body([])
    body.release = asyncio.Event()
    client.get_object.return_value = {"Body": body}
    original_create = file_storage.tempfile.NamedTemporaryFile
    original_unlink = Path.unlink
    started, release = threading.Event(), threading.Event()
    cleanup_attempted = threading.Event()
    handles = []

    def create(**kwargs):
        handle = original_create(dir=tmp_path, **kwargs)
        handles.append(handle)
        original_close = handle.close

        def close():
            original_close()
            if failure == "close":
                cleanup_attempted.set()
                raise PermissionError("cannot close download")

        handle.close = close
        started.set()
        if phase == "creation" and not release.wait(5):
            raise TimeoutError("creation not released")
        return handle

    def unlink(path, *args, **kwargs):
        if path.parent == tmp_path and failure == "unlink":
            cleanup_attempted.set()
            raise PermissionError("cannot remove download")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(file_storage.tempfile, "NamedTemporaryFile", create)
    monkeypatch.setattr(Path, "unlink", unlink)
    registry = MLClientRegistry(bm25_loader=lambda: persistence.load_bm25_index_from_s3(storage))
    operation = {
        "storage": lambda: storage.download_to_temp("report.pdf"),
        "loader": registry.ensure_bm25_loaded,
        "mutation": lambda: BM25IndexAdapter(registry).add("new", text_hash="new-hash"),
    }[consumer]()
    pending = asyncio.create_task(operation)
    try:
        if phase == "creation":
            assert await asyncio.to_thread(started.wait, 5)
        else:
            await asyncio.wait_for(body.started.wait(), 5)
        pending.cancel("request cancelled")
        release.set()
        with pytest.raises(asyncio.CancelledError, match="request cancelled"):
            await asyncio.wait_for(pending, 5)
        assert pending.cancelled()
        assert cleanup_attempted.is_set()
        assert handles[0].closed
        assert body.closed
        assert storage._active == 0
        with pytest.raises(persistence.BM25SnapshotNotReady):
            registry.bm25_index()
    finally:
        release.set()
        for handle in handles:
            original_unlink(Path(handle.name), missing_ok=True)
        await registry.close()
        await storage.aclose()


@pytest.mark.asyncio
async def test_cancelled_temp_creation_failure_preserves_cancellation(monkeypatch):
    storage, client, _, _ = client_harness(monkeypatch)
    body = Body([])
    client.get_object.return_value = {"Body": body}
    started, release = threading.Event(), threading.Event()

    def create(**kwargs):
        started.set()
        if not release.wait(5):
            raise TimeoutError("creation not released")
        raise PermissionError("cannot create download")

    monkeypatch.setattr(file_storage.tempfile, "NamedTemporaryFile", create)
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
