"""Document repository interface -- CRUD and status queries for Document entities."""

from __future__ import annotations

from typing import Protocol

from domain.entities.document import Document


class DocumentRepository(Protocol):
    async def save(self, document: Document) -> Document: ...
    async def get_by_id(self, document_id: int) -> Document | None: ...
    async def delete(self, document_id: int) -> None: ...
    async def update_status(
        self,
        document_id: int,
        status: str,
        error: str | None = None,
        chunks: int | None = None,
        chars: int | None = None,
        warning: str | None = None,
        quality_score: float | None = None,
    ) -> None: ...
    async def set_source_path(self, document_id: int, source_path: str) -> None: ...
    async def set_domain(self, document_id: int, doc_domain: str) -> None: ...
    async def find_active_slot(
        self, owner_id: int | None, filename: str, group_id: int | None, for_update: bool = False
    ) -> Document | None: ...

    async def list_visible(
        self,
        user_kind: str,
        user_id: int,
        group_ids: list[int],
        assigned_client_ids: list[int],
    ) -> list[Document]: ...
    async def list_all(self) -> list[Document]: ...
    async def set_has_manual_edits(self, document_id: int, value: bool) -> None: ...
    async def update_chunk_stats(self, document_id: int, chunks: int, chars: int) -> None: ...
    async def list_distinct_filenames(self, search: str | None = None, limit: int = 100) -> list[str]: ...
