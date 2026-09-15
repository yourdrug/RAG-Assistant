"""IngestionRegistry -- UoW-based registry for tracking indexed S3 files.

Extracted from IngestionService to isolate the "which files are already indexed"
concern.  The critical invariant: a registry row is only committed AFTER the
corresponding document/chunk sync transaction has committed.  A crash between
sync-commit and registry-upsert leaves no 'indexed' row, so the next
incremental run re-indexes the file (safe) instead of silently skipping it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from domain.repositories.ingestion_registry_repository import IngestionRegistryEntry

if TYPE_CHECKING:
    from application.ports.file_storage import FileItem, FileStorage
    from application.ports.unit_of_work_factory import UnitOfWorkFactory

log = logging.getLogger("default")


def s3_file_hash(file_item: "FileItem") -> str:
    return f"{file_item.size_bytes}_{file_item.last_modified}"


class IngestionRegistry:
    """Tracks which S3 files have been indexed via the ingestion UoW."""

    def __init__(self, uow_factory: "UnitOfWorkFactory", file_storage: "FileStorage") -> None:
        self._uow_factory = uow_factory
        self._file_storage = file_storage

    async def upsert(
        self,
        filename: str,
        file_hash_val: str,
        source: str,
        chunks_count: int,
        chars: int,
        indexed_at: datetime | None = None,
    ) -> None:
        async with self._uow_factory.create(master=True) as uow:
            entry = IngestionRegistryEntry(
                filename=filename,
                file_hash=file_hash_val,
                source=source,
                chunks=chunks_count,
                chars=chars,
                indexed_at=indexed_at or datetime.now(),
            )
            await uow.ingestion_registry.upsert(entry)

    async def is_indexed(self, filename: str, file_hash_val: str) -> bool:
        async with self._uow_factory.create(master=True) as uow:
            return await uow.ingestion_registry.is_already_indexed(filename, file_hash_val)

    async def list_all(self) -> dict:
        async with self._uow_factory.create(master=True) as uow:
            entries = await uow.ingestion_registry.list_all()
            return {
                name: {
                    "hash": e.file_hash,
                    "source": e.source,
                    "chunks": e.chunks,
                    "chars": e.chars,
                    "indexed_at": e.indexed_at,
                }
                for name, e in entries.items()
            }

    async def delete(self, filename: str) -> None:
        async with self._uow_factory.create(master=True) as uow:
            await uow.ingestion_registry.delete(filename)

    def collect_entries(
        self,
        docs: list,
        chunks: list,
        source_chars: dict[str, int],
    ) -> dict[str, dict]:
        """Build registry entries in memory (no DB write).

        The registry row must only be committed AFTER the document/chunk
        sync transaction.
        """
        entries: dict[str, dict] = {}
        for src, chars in source_chars.items():
            fname = Path(src).name
            key = "/".join(src.split("/")[3:])
            file_info = self._file_storage.get_file_info(key)
            h = s3_file_hash(file_info) if file_info else "unknown"
            chunks_count = sum(1 for c in chunks if c.metadata.get("source") == src)
            entries[fname] = {
                "hash": h,
                "source": src,
                "chunks": chunks_count,
                "chars": chars,
            }
        return entries

    async def persist_entries(self, entries: dict[str, dict]) -> None:
        """Upsert collected registry entries (call after the sync committed)."""
        for fname, info in entries.items():
            await self.upsert(
                fname,
                info["hash"],
                info["source"],
                info["chunks"],
                info["chars"],
                indexed_at=datetime.now(),
            )
