"""Persistence and search ports for document chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

    from domain.knowledge.chunk import ChunkInsert, ChunkRecord, ChunkStats
    from domain.knowledge.document_nodes import DocumentNode
    from domain.knowledge.revision import RevisionSwitch, SourceVersion, StagedRevision
    from domain.shared.pagination import CursorPage


@dataclass(frozen=True)
class ChunkInsertOptions:
    owner_id: int | None = None
    group_id: int | None = None
    # The document's space, copied to its chunks as a search selection label.
    space_id: int | None = None
    doc_domain: str = "general"
    domain_metadata: dict[str, Any] | None = None
    pack_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ManualChunkInsertOptions:
    owner_id: int | None = None
    group_id: int | None = None
    space_id: int | None = None
    manual: bool = False
    content_hash: str | None = None
    context_metadata: dict[str, Any] | None = None
    embed_text: str | None = None
    doc_title: str | None = None
    pack_metadata: dict[str, Any] | None = None


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

    async def set_acl(
        self, document_id: int, visibility: str, owner_id: int | None, group_id: int | None
    ) -> None:
        """Copy the document's ACL to its chunks, staged ones included."""
        ...


@runtime_checkable
class ChunkCorpusRepository(Protocol):
    """Read the chunk corpus for sparse-index rebuilding."""

    def iter_all_contents_with_acl(
        self, batch_size: int = 1000
    ) -> AsyncIterator[tuple[str, str, int | None, int | None]]: ...

    async def list_space_filenames(self, space_id: int) -> dict[int, str]:
        """File name of every document of the space that has chunks, by document id."""
        ...

    async def list_document_chunks(self, document_ids: list[int]) -> list[ChunkRecord]:
        """All chunks of the documents, in document and chunk order."""
        ...


@runtime_checkable
class ChunkPackMetadataRepository(Protocol):
    """Pack-owned metadata is per document: every chunk of a document carries the same value."""

    async def set_pack_metadata(self, document_id: int, pack_metadata: dict[str, Any]) -> int: ...

    async def get_pack_metadata(self, document_id: int) -> dict[str, Any] | None:
        """Pack metadata shared by the document's chunks; ``None`` when it has no chunks."""
        ...


@runtime_checkable
class DocumentNodeRepository(Protocol):
    """Parent scopes of a document's chunks, replaced together with the chunks."""

    async def replace_document_nodes(self, document_id: int, nodes: list[DocumentNode]) -> None: ...


@runtime_checkable
class ChunkRevisionRepository(Protocol):
    """Hidden (staged) revision of a document's chunks and its publication.

    Staged chunks take ids from the same sequence as published chunks, so a point
    keeps its id when its revision is published. Every method expects the caller
    to hold the document row lock.
    """

    async def stage_revision(
        self,
        document_id: int,
        filename: str,
        visibility: str,
        chunks: list[ChunkInsert],
        *,
        nodes: list[DocumentNode],
        options: ChunkInsertOptions = DEFAULT_CHUNK_INSERT_OPTIONS,
        source_version: SourceVersion | None = None,
    ) -> StagedRevision:
        """Write a new staged revision; an existing one must be discarded first.

        ``source_version`` travels with the revision and is returned on publication.
        """
        ...

    async def get_staged_revision(self, document_id: int) -> StagedRevision | None: ...

    async def staged_source_versions(self, document_ids: list[int]) -> dict[int, SourceVersion]:
        """Source versions of the staged revisions of these documents, where one is recorded."""
        ...

    async def get_staged_by_ids(
        self, document_id: int, token: str, chunk_ids: list[int]
    ) -> list[ChunkRecord]:
        """Staged chunks of this revision token; empty once it is published or discarded."""
        ...

    async def discard_staged_revision(self, document_id: int) -> list[int]:
        """Remove the staged revision; returns its chunk ids for vector cleanup."""
        ...

    async def publish_staged_revision(self, document_id: int, token: str) -> RevisionSwitch | None:
        """Replace the published chunks and nodes with the staged revision of this token.

        Returns ``None`` when the token is no longer the document's staged revision.
        """
        ...
