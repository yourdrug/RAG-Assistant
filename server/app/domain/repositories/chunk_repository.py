"""Chunk repository interface -- exact substring search (pg_trgm) for document chunks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ChunkSearchResult:
    """A single chunk match from substring search."""

    chunk_id: int
    document_id: int
    filename: str
    content: str
    chunk_index: int


class ChunkRepository(ABC):
    """Interface for chunk storage used by pg_trgm substring search."""

    @abstractmethod
    async def bulk_insert(
        self,
        document_id: int,
        filename: str,
        visibility: str,
        chunks: list[str],
        owner_id: int | None = None,
        group_id: int | None = None,
        doc_domain: str = "general",
    ) -> None:
        """Insert chunks for a document. Replaces existing chunks for this document."""

    @abstractmethod
    async def search_substring(
        self,
        query: str,
        user: dict,
        group_ids: list[int],
        assigned_client_ids: list[int],
        limit: int = 20,
        mode: str = "exact",
    ) -> list[ChunkSearchResult]:
        """Search chunks by substring. mode='exact'=pg_trgm ranked, 'icontains'=plain ILIKE."""

    @abstractmethod
    async def delete_by_document_id(self, document_id: int) -> None:
        """Delete all chunks for a document."""
