"""Chunk repository interface -- CRUD, search, and versioning protocols for document chunks.

The combined ``ChunkRepository`` inherits from three narrow protocols so that
the UnitOfWork can expose a single attribute while individual services declare
only the subset they actually depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, runtime_checkable

from domain.value_objects.cursor_page import CursorPage
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
    content_hash: str | None = None
    section: str | None = None
    heading: str | None = None
    heading_level: int | None = None
    content_type: str | None = None
    doc_title: str | None = None
    doc_type: str | None = None


@dataclass(frozen=True)
class ChunkStats:
    """Aggregate statistics for chunks of a document."""

    total_chunks: int
    total_chars: int


@runtime_checkable
class ChunkCrudRepository(Protocol):
    """CRUD operations for document chunks."""

    async def bulk_insert(
        self,
        document_id: int,
        filename: str,
        visibility: str,
        chunks: list[str],
        owner_id: int | None = None,
        group_id: int | None = None,
        doc_domain: str = DocDomain.GENERAL.value,
        content_hashes: list[str] | None = None,
        domain_metadata: dict | None = None,
        act_version_id: int | None = None,
        effective_from: datetime | None = None,
        effective_to: datetime | None = None,
        is_current: bool = True,
        sections: list[str | None] | None = None,
        headings: list[str | None] | None = None,
        heading_levels: list[int | None] | None = None,
        content_types: list[str | None] | None = None,
        doc_titles: list[str | None] | None = None,
        doc_types: list[str | None] | None = None,
    ) -> list[int]: ...

    async def get_by_id(self, chunk_id: int) -> ChunkSearchResult | None: ...

    async def get_max_chunk_index(self, document_id: int) -> int: ...

    async def update_content(
        self,
        chunk_id: int,
        content: str,
        edited_at: datetime,
        edited_by: int,
    ) -> None: ...

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
        content_hash: str | None = None,
    ) -> int: ...

    async def delete_one(self, chunk_id: int) -> None: ...

    async def delete_by_document_id(self, document_id: int) -> None: ...

    async def list_for_document(
        self,
        document_id: int,
        limit: int = 50,
        offset: int = 0,
        content_hashes: list[str] | None = None,
    ) -> tuple[list[ChunkSearchResult], int]: ...

    async def list_for_document_cursor(
        self,
        document_id: int,
        limit: int = 50,
        cursor: tuple[int, int] | None = None,
        direction: str = "next",
        content_hashes: list[str] | None = None,
    ) -> CursorPage[ChunkSearchResult]: ...

    async def find_duplicate_by_hash(
        self,
        document_id: int,
        content_hash: str,
        exclude_chunk_id: int | None = None,
    ) -> ChunkSearchResult | None: ...

    async def get_document_stats(self, document_id: int) -> ChunkStats: ...

    async def get_all_contents(self) -> list[str]: ...

    async def get_all_contents_batches(self, batch_size: int = 5000) -> list[list[str]]:
        """Yield batches of chunk contents to avoid loading all into memory."""
        ...

    async def get_neighbors(
        self,
        document_id: int,
        center_index: int,
        window: int = 1,
        exclude_hashes: set[str] | None = None,
        user: dict | None = None,
        group_ids: list[int] | None = None,
    ) -> list[ChunkSearchResult]: ...

    async def get_table_batches(
        self,
        document_id: int,
        anchor_index: int,
        exclude_hashes: set[str] | None = None,
        user: dict | None = None,
        group_ids: list[int] | None = None,
    ) -> list[ChunkSearchResult]: ...

    async def update_filename_by_document_id(self, document_id: int, new_filename: str) -> int: ...


@runtime_checkable
class ChunkSearchRepository(Protocol):
    """Substring search over document chunks."""

    async def search_substring(
        self,
        query: str,
        user: dict,
        group_ids: list[int],
        limit: int = 20,
        mode: str = SearchMode.EXACT.value,
        document_id: int | None = None,
        managed_client_ids: list[int] | None = None,
        managed_internal_ids: list[int] | None = None,
        managed_group_ids: list[int] | None = None,
    ) -> list[ChunkSearchResult]: ...


@runtime_checkable
class ChunkVersioningRepository(Protocol):
    """Temporal versioning operations for act-versioned chunks."""

    async def set_current_by_act_version_ids(self, act_version_ids: list[int], is_current: bool) -> int: ...

    async def update_temporal_by_act_version_id(
        self,
        act_version_id: int,
        effective_from: date | None,
        effective_to: date | None,
    ) -> int: ...


@runtime_checkable
class ChunkRepository(ChunkCrudRepository, ChunkSearchRepository, ChunkVersioningRepository, Protocol):
    """Combined protocol -- the UnitOfWork attribute type.

    Services should prefer the narrow protocols (``ChunkCrudRepository``,
    ``ChunkSearchRepository``, ``ChunkVersioningRepository``) for their
    constructor type hints.
    """

    ...
