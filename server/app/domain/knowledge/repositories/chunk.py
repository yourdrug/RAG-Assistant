"""Persistence and search ports for document chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import date, datetime

    from domain.knowledge.chunk import ChunkInsert, ChunkRecord, ChunkStats
    from domain.knowledge.document_nodes import DocumentNode
    from domain.shared.pagination import CursorPage


@dataclass(frozen=True)
class ChunkInsertOptions:
    owner_id: int | None = None
    group_id: int | None = None
    doc_domain: str = "general"
    domain_metadata: dict[str, Any] | None = None
    act_version_id: int | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    is_current: bool = True


@dataclass(frozen=True)
class ManualChunkInsertOptions:
    owner_id: int | None = None
    group_id: int | None = None
    manual: bool = False
    content_hash: str | None = None
    context_metadata: dict[str, Any] | None = None
    embed_text: str | None = None
    doc_title: str | None = None
    act_version_id: int | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    is_current: bool = True


DEFAULT_CHUNK_INSERT_OPTIONS = ChunkInsertOptions()
DEFAULT_MANUAL_CHUNK_INSERT_OPTIONS = ManualChunkInsertOptions()


@runtime_checkable
class ChunkCrudRepository(Protocol):
    """CRUD operations for document chunks."""

    async def get_ids_by_document_id(self, document_id: int, *, for_update: bool = False) -> list[int]:
        """Read chunk IDs; optionally lock the document until the transaction ends."""
        ...

    async def bulk_insert(
        self,
        document_id: int,
        filename: str,
        visibility: str,
        chunks: list[ChunkInsert],
        options: ChunkInsertOptions = DEFAULT_CHUNK_INSERT_OPTIONS,
    ) -> list[int]: ...

    async def get_by_id(self, chunk_id: int) -> ChunkRecord | None: ...

    async def get_by_ids(self, document_id: int, chunk_ids: list[int]) -> list[ChunkRecord]: ...

    async def get_max_chunk_index(self, document_id: int) -> int: ...

    async def update_content(
        self,
        chunk_id: int,
        content: str,
        content_hash: str,
        edited_at: datetime,
        edited_by: int,
        *,
        embed_text: str | None = None,
    ) -> None: ...

    async def insert_one(
        self,
        document_id: int,
        chunk_index: int,
        content: str,
        filename: str,
        visibility: str,
        *,
        doc_domain: str,
        options: ManualChunkInsertOptions = DEFAULT_MANUAL_CHUNK_INSERT_OPTIONS,
    ) -> int: ...

    async def delete_one(self, chunk_id: int) -> None: ...

    async def delete_by_document_id(self, document_id: int) -> None: ...

    async def list_for_document(
        self, document_id: int, limit: int = 50, offset: int = 0, content_hashes: list[str] | None = None
    ) -> tuple[list[ChunkRecord], int]: ...

    async def list_for_document_cursor(
        self,
        document_id: int,
        limit: int = 50,
        cursor: tuple[int, int] | None = None,
        direction: str = "next",
        content_hashes: list[str] | None = None,
    ) -> CursorPage[ChunkRecord]: ...

    async def find_duplicate_by_hash(
        self, document_id: int, content_hash: str, exclude_chunk_id: int | None = None
    ) -> ChunkRecord | None: ...

    async def get_document_stats(self, document_id: int) -> ChunkStats: ...

    async def update_filename_by_document_id(self, document_id: int, new_filename: str) -> int: ...


@runtime_checkable
class ChunkCorpusRepository(Protocol):
    """Read the chunk corpus for sparse-index rebuilding."""

    def iter_all_contents_with_acl(
        self, batch_size: int = 1000
    ) -> AsyncIterator[tuple[str, str, int | None, int | None]]: ...


@runtime_checkable
class ChunkVersioningRepository(Protocol):
    """Temporal versioning operations for act-versioned chunks."""

    async def set_current_by_act_version_ids(
        self, act_version_ids: list[int], *, is_current: bool
    ) -> int: ...

    async def update_temporal_by_act_version_id(
        self, act_version_id: int, effective_from: date | None, effective_to: date | None
    ) -> int: ...


@runtime_checkable
class DocumentNodeRepository(Protocol):
    """Parent scopes of a document's chunks, replaced together with the chunks."""

    async def replace_document_nodes(
        self, document_id: int, act_version_id: int | None, nodes: list[DocumentNode]
    ) -> None: ...
