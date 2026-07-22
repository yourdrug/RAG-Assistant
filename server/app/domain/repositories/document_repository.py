"""Document Repository interface."""

from __future__ import annotations

from typing import Protocol

from domain.entities.document import Document


class DocumentRepository(Protocol):
    def save(self, document: Document) -> Document: ...
    def get_by_id(self, document_id: int) -> Document | None: ...
    def delete(self, document_id: int) -> None: ...
    def update_status(
        self,
        document_id: int,
        status: str,
        error: str | None = None,
        chunks: int | None = None,
        chars: int | None = None,
    ) -> None: ...
    def set_source_path(self, document_id: int, source_path: str) -> None: ...
    def find_active_slot(
        self, owner_id: int | None, filename: str, group_id: int | None
    ) -> Document | None: ...
    def list_visible(
        self,
        user_kind: str,
        user_id: int,
        group_ids: list[int],
        assigned_client_ids: list[int],
    ) -> list[Document]: ...
