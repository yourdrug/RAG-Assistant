"""Document repository interface -- CRUD and maintenance protocols for Document entities.

The combined ``DocumentRepository`` inherits from two narrow protocols so that
the UnitOfWork can expose a single attribute while scheduler/health services
depend only on the maintenance subset.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from domain.entities.document import Document


@runtime_checkable
class DocumentCrudRepository(Protocol):
    """CRUD and query operations for Document entities."""

    async def save(self, document: Document) -> Document: ...
    async def get_by_id(self, document_id: int) -> Document | None: ...
    async def delete(self, document_id: int) -> None: ...
    async def update_status(
        self,
        document_id: int,
        status: str | None = None,
        error: str | None = None,
        chunks: int | None = None,
        chars: int | None = None,
        warning: str | None = None,
        quality_score: float | None = None,
    ) -> None: ...
    async def set_source_path(self, document_id: int, source_path: str) -> None: ...
    async def set_domain(self, document_id: int, doc_domain: str) -> None: ...
    async def find_active_slot(
        self,
        owner_id: int | None,
        filename: str,
        group_id: int | None,
        for_update: bool = False,
    ) -> Document | None: ...
    async def find_active_slots_by_filenames(self, filenames: list[str]) -> list[Document]: ...
    async def list_visible(
        self,
        user_kind: str,
        user_id: int,
        group_ids: list[int],
        user_role: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Document]: ...
    async def list_all(self, limit: int = 200, offset: int = 0) -> list[Document]: ...
    async def list_warned(self, quality_threshold: float = 0.3) -> list[Document]: ...
    async def set_has_manual_edits(self, document_id: int, value: bool) -> None: ...
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

    ...
