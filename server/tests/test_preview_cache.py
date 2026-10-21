"""Preview expiration follows the explicitly supplied TTL."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from application.documents.preview_cache import PreviewCache
from application.ports.file_storage import FileStorage


@pytest.mark.asyncio
async def test_preview_expires_at_configured_ttl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clock = Mock(return_value=100.0)
    monkeypatch.setattr("application.documents.preview_cache.time.monotonic", clock)
    storage = AsyncMock(spec=FileStorage)
    cache = PreviewCache(storage, settings=SimpleNamespace(preview_cache_ttl_seconds=10))
    preview_id = await cache.store(b"pdf", original_filename="document.pdf")
    key = storage.upload_file.await_args.args[0]
    preview = tmp_path / "preview.pdf"
    preview.write_bytes(b"pdf")
    storage.download_to_temp.return_value = preview

    clock.return_value = 109.0
    assert await cache.get_bytes(preview_id) == b"pdf"
    storage.delete_file.assert_not_awaited()

    clock.return_value = 111.0
    assert await cache.get_bytes(preview_id) is None
    storage.delete_file.assert_awaited_once_with(key)
    assert storage.download_to_temp.await_count == 1
