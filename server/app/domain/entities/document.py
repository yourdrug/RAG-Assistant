"""Document Entity — Aggregate Root for Knowledge Base context."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserRole
from domain.value_objects.visibility import DocumentVisibility


@dataclass
class Document:
    id: int | None = None
    filename: str = ""
    source_path: str = ""
    visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC
    owner_id: int | None = None
    group_id: int | None = None
    status: DocumentStatus = DocumentStatus.PENDING
    error_message: str | None = None
    chunks: int | None = None
    chars: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    indexed_at: datetime | None = None

    def __post_init__(self) -> None:
        if isinstance(self.visibility, str):
            self.visibility = DocumentVisibility.validate(self.visibility)
        if isinstance(self.status, str):
            self.status = DocumentStatus(self.status)

    def mark_processing(self) -> None:
        self.status = DocumentStatus.PROCESSING

    def mark_done(self, chunks: int, chars: int) -> None:
        self.status = DocumentStatus.DONE
        self.chunks = chunks
        self.chars = chars
        self.indexed_at = datetime.now(UTC)

    def mark_failed(self, error: str) -> None:
        self.status = DocumentStatus.FAILED
        self.error_message = error

    def can_be_deleted_by(
        self, user_id: int, user_role: UserRole, user_group_ids: list[int] | None = None
    ) -> bool:
        if user_role == UserRole.ADMIN:
            return True
        if self.owner_id == user_id:
            return True
        if (
            self.visibility == DocumentVisibility.INTERNAL_GROUP
            and self.group_id is not None
            and user_group_ids is not None
            and self.group_id in user_group_ids
        ):
            return True
        return False
