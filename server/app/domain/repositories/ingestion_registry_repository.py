"""IngestionRegistry repository protocol — tracks which files have been indexed.

Replaces the JSON-file based registry with a Postgres table for
concurrent access and durability across container restarts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class IngestionRegistryEntry:
    filename: str
    file_hash: str
    source: str
    chunks: int
    chars: int
    indexed_at: str | None = None


@runtime_checkable
class IngestionRegistryRepository(Protocol):
    async def get(self, filename: str) -> IngestionRegistryEntry | None: ...

    async def upsert(self, entry: IngestionRegistryEntry) -> None: ...

    async def delete(self, filename: str) -> None: ...

    async def list_all(self) -> dict[str, IngestionRegistryEntry]: ...

    async def is_already_indexed(self, filename: str, file_hash: str) -> bool: ...
