"""Document repository interface -- CRUD and maintenance protocols for Document entities.

The combined ``DocumentRepository`` inherits from two narrow protocols so that
the UnitOfWork can expose a single attribute while scheduler/health services
depend only on the maintenance subset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict, Unpack, runtime_checkable


if TYPE_CHECKING:
    from domain.knowledge.document import Document
    from domain.knowledge.enums import DocumentVisibility


class DocumentStatusUpdate(TypedDict, total=False):
    status: str | None
    error: str | None
    chunks: int | None
    chars: int | None
    warning: str | None
    quality_score: float | None
    expected_version: int | None


class DocumentListOptions(TypedDict, total=False):
    user_role: str | None
    limit: int
    offset: int
    managed_client_ids: list[int] | None
    managed_internal_ids: list[int] | None
    managed_group_ids: list[int] | None


@runtime_checkable
class DocumentCrudRepository(Protocol):
    """CRUD and query operations for Document entities."""

    async def save(self, document: Document) -> Document: ...
    async def get_by_id(self, document_id: int, *, for_update: bool = False) -> Document | None: ...
    async def find_ingestion_documents(self, keys: list[str]) -> list[Document]: ...
    async def list_ingestion_documents(self, prefix: str) -> list[Document]: ...
    async def get_by_ids(self, document_ids: list[int]) -> list[Document]: ...
    async def delete(self, document_id: int) -> None: ...
    async def update_status(
        self,
        document_id: int,
        **changes: Unpack[DocumentStatusUpdate],
    ) -> bool: ...
    async def set_source_path(self, document_id: int, source_path: str) -> None: ...
    async def set_domain(self, document_id: int, doc_domain: str) -> None: ...
    async def set_acl(
        self,
        document_id: int,
        visibility: DocumentVisibility,
        owner_id: int | None,
        group_id: int | None,
    ) -> None:
        """Lock the document and update ACL within the caller's transaction."""
        ...

    async def find_active_slot(
        self,
        owner_id: int | None,
        filename: str,
        group_id: int | None,
        *,
        for_update: bool = False,
    ) -> Document | None: ...
    async def find_active_slots_by_filenames(
        self, filenames: list[str], *, owner_id: int | None = None, group_id: int | None = None
    ) -> list[Document]: ...
    async def list_visible(
        self,
        user_kind: str,
        user_id: int,
        group_ids: list[int],
        **options: Unpack[DocumentListOptions],
    ) -> list[Document]: ...
    async def list_all(self, limit: int = 200, offset: int = 0) -> list[Document]: ...
    async def list_warned(self, quality_threshold: float = 0.3) -> list[Document]: ...
    async def set_has_manual_edits(self, document_id: int, *, value: bool) -> None: ...
    async def update_chunk_stats(self, document_id: int, chunks: int, chars: int) -> None: ...
    async def list_distinct_filenames(self, search: str | None = None, limit: int = 100) -> list[str]: ...
    async def update_filename(self, document_id: int, new_filename: str, new_source_path: str) -> None: ...
    async def mark_done_if_indexing(self, document_id: int) -> bool: ...


@runtime_checkable
class DocumentMaintenanceRepository(Protocol):
    """Self-healing and scheduler operations (not part of normal CRUD lifecycle)."""

    async def delete_internal_documents(self) -> int: ...
    async def mark_stuck_processing_failed(self) -> list[int]: ...
    async def reconcile_indexing_documents(self) -> list[int]: ...


@runtime_checkable
class DocumentRepository(DocumentCrudRepository, DocumentMaintenanceRepository, Protocol):
    """Combined protocol -- the UnitOfWork attribute type.

    Services should prefer the narrow protocols for their constructor type hints.
    """
