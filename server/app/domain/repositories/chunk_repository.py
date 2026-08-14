"""Chunk repository interface -- exact substring search (pg_trgm) for document chunks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ChunkSearchResult:
    """A single chunk match from substring search."""

    chunk_id: int
    document_id: int
    filename: str
    content: str
    chunk_index: int
    visibility: str = ""
    doc_domain: str = "general"
    owner_id: int | None = None
    group_id: int | None = None
    edited_at: datetime | None = None
    edited_by: int | None = None
    manual: bool = False
    creation_date: datetime | None = None


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
    async def get_by_id(self, chunk_id: int) -> ChunkSearchResult | None:
        """Get a single chunk by its ID."""

    @abstractmethod
    async def get_max_chunk_index(self, document_id: int) -> int:
        """Get the maximum chunk_index for a document. Returns -1 if no chunks exist."""

    @abstractmethod
    async def update_content(
        self,
        chunk_id: int,
        content: str,
        edited_at: datetime,
        edited_by: int,
    ) -> None:
        """Update chunk content and mark as edited."""

    @abstractmethod
    async def insert_one(
        self,
        document_id: int,
        chunk_index: int,
        content: str,
        filename: str,
        visibility: str,
        doc_domain: str,
        owner_id: int | None = None,
        group_id: int | None = None,
        manual: bool = False,
    ) -> int:
        """Insert a single chunk. Returns the generated chunk_id."""

    @abstractmethod
    async def delete_one(self, chunk_id: int) -> None:
        """Delete a single chunk by ID."""

    @abstractmethod
    async def search_substring(
        self,
        query: str,
        user: dict,
        group_ids: list[int],
        assigned_client_ids: list[int],
        limit: int = 20,
        mode: str = "exact",
        document_id: int | None = None,
    ) -> list[ChunkSearchResult]:
        """Search chunks by substring. mode='exact'=pg_trgm ranked, 'icontains'=plain ILIKE."""

    @abstractmethod
    async def delete_by_document_id(self, document_id: int) -> None:
        """Delete all chunks for a document."""
