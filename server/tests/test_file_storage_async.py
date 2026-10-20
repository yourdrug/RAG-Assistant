"""Async metadata contract and event-loop responsiveness."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.ingestion import IngestionRegistry
from infrastructure.storage.file_storage import LocalStorage


@pytest.mark.asyncio
@pytest.mark.parametrize("method,args", [("list_files", ("docs",)), ("get_file_info", ("docs/a.txt",))])
async def test_list_and_head_yield_to_other_coroutines(tmp_path, monkeypatch, method, args):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/a.txt").write_text("content")
    storage = LocalStorage(str(tmp_path))
    started, release = threading.Event(), threading.Event()
    original = type(tmp_path).stat

    def delayed_stat(path, *args, **kwargs):
        if path == tmp_path / args_key:
            started.set()
            if not release.wait(5):
                raise TimeoutError("metadata blocked event loop")
        return original(path, *args, **kwargs)

    args_key = args[0]
    monkeypatch.setattr(type(tmp_path), "stat", delayed_stat)
    task = asyncio.create_task(getattr(storage, method)(*args))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        assert not task.done()
    finally:
        release.set()
    result = await asyncio.wait_for(task, 5)
    assert result
    await storage.aclose()


@pytest.mark.asyncio
async def test_collect_entries_awaits_metadata_and_preserves_hash_and_counts():
    info = MagicMock(size_bytes=42, last_modified="2026-01-01T00:00:00+00:00")
    storage = MagicMock(get_file_info=AsyncMock(return_value=info))
    registry = IngestionRegistry(MagicMock(), storage)
    source = "s3://bucket/docs/отчёт.txt"
    chunks = [MagicMock(metadata={"source": source}) for _ in range(2)]
    entries = await registry.collect_entries([], chunks, {source: 80})
    assert entries == {
        "docs/отчёт.txt": {
            "filename": "отчёт.txt",
            "hash": "42_2026-01-01T00:00:00+00:00",
            "source": source,
            "chunks": 2,
            "chars": 80,
        }
    }
    storage.get_file_info.assert_awaited_once_with("docs/отчёт.txt")


@pytest.mark.asyncio
async def test_local_report_stream_cleans_partial_on_exception_and_cancellation(tmp_path):
    import asyncio
    from infrastructure.storage.file_storage import LocalStorage

    storage = LocalStorage(str(tmp_path))
    await storage.upload_file('report.json', b'original')
    for error in (OSError('failed'), asyncio.CancelledError()):

        async def chunks(error=error):
            yield b'partial'
            raise error

        with pytest.raises(type(error)):
            await storage.upload_stream('report.json', chunks())
        assert (tmp_path / 'report.json').read_bytes() == b'original'
        assert list(tmp_path.iterdir()) == [tmp_path / 'report.json']
