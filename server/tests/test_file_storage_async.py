"""Async metadata contract and event-loop responsiveness."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from application.ingestion import IngestionRegistry


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
