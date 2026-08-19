"""Chunk repository interface -- exact substring search (pg_trgm) for document chunks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.search_mode import SearchMode


@dataclass
class ChunkSearchResult:
    """A single chunk match from substring search."""

    chunk_id: int
    document_id: int
    filename: str
    content: str
    chunk_index: int
    visibility: str = ""
    doc_domain: str = DocDomain.GENERAL.value
    owner_id: int | None = None
    group_id: int | None = None
    edited_at: datetime | None = None
    edited_by: int | None = None
    manual: bool = False
    creation_date: datetime | None = None


@dataclass(frozen=True)
class ChunkStats:
    """Aggregate statistics for chunks of a document."""

    total_chunks: int
    total_chars: int


@runtime_checkable
class ChunkRepository(Protocol):
    """Interface for chunk storage used by pg_trgm substring search."""

    async def bulk_insert(
        self,
        document_id: int,
        filename: str,
        visibility: str,
        chunks: list[str],
        owner_id: int | None = None,
        group_id: int | None = None,
        doc_domain: str = DocDomain.GENERAL.value,
    ) -> None:
        """Insert chunks for a document. Replaces existing chunks for this document."""
        ...

    async def get_by_id(self, chunk_id: int) -> ChunkSearchResult | None:
        """Get a single chunk by its ID."""
        ...

    async def get_max_chunk_index(self, document_id: int) -> int:
        """Get the maximum chunk_index for a document. Returns -1 if no chunks exist."""
        ...

    async def update_content(
        self,
        chunk_id: int,
        content: str,
        edited_at: datetime,
        edited_by: int,
    ) -> None:
        """Update chunk content and mark as edited."""
        ...

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
        ...

    async def delete_one(self, chunk_id: int) -> None:
        """Delete a single chunk by ID."""
        ...

    async def search_substring(
        self,
        query: str,
        user: dict,
        group_ids: list[int],
        limit: int = 20,
        mode: str = SearchMode.EXACT.value,
        document_id: int | None = None,
    ) -> list[ChunkSearchResult]:
        """Search chunks by substring. mode='exact'=pg_trgm ranked, 'icontains'=plain ILIKE."""
        ...

    async def delete_by_document_id(self, document_id: int) -> None:
        """Delete all chunks for a document."""
        ...

    async def list_for_document(
        self, document_id: int, limit: int = 50, offset: int = 0
    ) -> tuple[list[ChunkSearchResult], int]:
        """List chunks for a document with pagination. Returns (chunks, total_count)."""
        ...

    async def find_duplicate_by_hash(
        self, document_id: int, content_hash: str, exclude_chunk_id: int | None = None
    ) -> ChunkSearchResult | None:
        """Find a chunk in the same document whose content matches the given hash."""
        ...

    async def get_document_stats(self, document_id: int) -> ChunkStats:
        """Return aggregate chunk count and total character length for a document."""
        ...

    async def get_all_contents(self) -> list[str]:
        """Return all chunk contents ordered by document_id, chunk_index (for BM25 rebuild)."""
        ...
