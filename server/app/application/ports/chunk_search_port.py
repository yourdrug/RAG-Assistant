"""Chunk search port — protocol for exact substring search used by RagService."""

from __future__ import annotations

from typing import Protocol

from domain.repositories.chunk_repository import ChunkSearchResult


class ChunkSearchPort(Protocol):
    """Protocol for searching chunks by exact substring (pg_trgm)."""

    async def search_substring(
        self,
        query: str,
        user: dict,
        group_ids: list[int],
        assigned_client_ids: list[int],
        limit: int = 20,
        mode: str = "exact",
    ) -> list[ChunkSearchResult]: ...
