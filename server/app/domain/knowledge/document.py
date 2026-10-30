"""Document entity -- aggregate root for the Knowledge Base bounded context.

Represents an uploaded file with its metadata (visibility, owner, status,
chunk count).  Owns lifecycle transitions (pending -> processing -> done/failed)
and ACL enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from domain.identity.roles import UserRole
from domain.knowledge.enums import (
    GENERAL_DOMAIN_KEY,
    DocumentStatus,
    DocumentVisibility,
    DomainKey,
    SourceType,
)
from domain.shared.exceptions import BusinessRuleViolationError


@dataclass(frozen=True)
class DocumentProcessingUpdate:
    status: DocumentStatus | None = None
    error: str | None = None
    chunks: int | None = None
    chars: int | None = None
    warning: str | None = None
    quality_score: float | None = None


@dataclass
class Document:
    id: int | None = None
    filename: str = ""
    source_path: str = ""
    ingestion_key: str | None = None
    visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC
    owner_id: int | None = None
    group_id: int | None = None
    status: DocumentStatus = DocumentStatus.PENDING
    doc_domain: DomainKey = GENERAL_DOMAIN_KEY
    source_type: str = SourceType.FILE.value
    has_manual_edits: bool = False
    version_group_id: int | None = None  # anchor document of the version chain
    version: int = 0  # optimistic locking counter
    error_message: str | None = None
    warning_message: str | None = None
    quality_score: float | None = None
    chunks: int | None = None
    chars: int | None = None
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))
    indexed_at: datetime | None = None

    def __post_init__(self) -> None:
        self.doc_domain = DomainKey(self.doc_domain)

        if isinstance(self.visibility, str):
            self.visibility = DocumentVisibility.parse(self.visibility)

        if isinstance(self.status, str):
            self.status = DocumentStatus(self.status)

    def apply_processing_update(self, change: DocumentProcessingUpdate) -> None:
        if change.status == DocumentStatus.DONE:
            self.mark_done(
                change.chunks if change.chunks is not None else self.chunks or 0,
                change.chars if change.chars is not None else self.chars or 0,
                change.warning,
                change.quality_score,
            )
        elif change.status == DocumentStatus.FAILED:
            self.mark_failed(change.error or "Document processing failed")
        elif change.status is not None:
            self.status = change.status

        self.apply_processing_metadata(change)

    def apply_processing_metadata(self, change: DocumentProcessingUpdate) -> None:
        if change.status != DocumentStatus.FAILED or change.error is not None:
            self.error_message = change.error

        if change.warning is not None:
            self.warning_message = change.warning

        if change.quality_score is not None:
            self.quality_score = change.quality_score

        if change.chunks is not None:
            self.chunks = change.chunks

        if change.chars is not None:
            self.chars = change.chars

    def mark_done(
        self, chunks: int, chars: int, warning_message: str | None = None, quality_score: float | None = None
    ) -> None:
        if not self.status.is_active:
            msg = f"Cannot mark document as DONE: status is {self.status.value!r} (expected active state)"
            raise BusinessRuleViolationError(msg)

        self.status = DocumentStatus.DONE
        self.chunks = chunks
        self.chars = chars
        self.indexed_at = datetime.now(UTC)
        self.warning_message = warning_message
        self.quality_score = quality_score

    def mark_failed(self, error: str) -> None:
        if not self.status.is_active:
            msg = f"Cannot mark document as FAILED: status is {self.status.value!r} (expected active state)"
            raise BusinessRuleViolationError(msg)

        self.status = DocumentStatus.FAILED
        self.error_message = error

    def in_curator_scope(
        self,
        user_group_ids: list[int] | None = None,
        *,
        managed_client_ids: list[int] | None = None,
        managed_internal_ids: list[int] | None = None,
        managed_group_ids: list[int] | None = None,
    ) -> bool:
        """Check assigned owners and groups; callers enforce the curator role."""
        all_managed_ids = list(set(managed_client_ids or []) | set(managed_internal_ids or []))

        if self.owner_id is not None and self.owner_id in all_managed_ids:
            return True

        all_group_ids = list(set(user_group_ids or []) | set(managed_group_ids or []))

        return bool(
            self.visibility == DocumentVisibility.INTERNAL_GROUP
            and self.group_id is not None
            and self.group_id in all_group_ids
        )

    def can_be_deleted_by(
        self,
        user_id: int,
        user_role: UserRole,
        user_group_ids: list[int] | None = None,
        *,
        managed_client_ids: list[int] | None = None,
        managed_internal_ids: list[int] | None = None,
        managed_group_ids: list[int] | None = None,
    ) -> bool:
        """Deleting a document requires the same mutation scope as editing its chunks."""
        return self.can_edit_chunks(
            user_id,
            user_role,
            user_group_ids,
            managed_client_ids=managed_client_ids,
            managed_internal_ids=managed_internal_ids,
            managed_group_ids=managed_group_ids,
        )

    def can_edit_chunks(
        self,
        user_id: int,
        user_role: UserRole,
        user_group_ids: list[int] | None = None,
        *,
        managed_client_ids: list[int] | None = None,
        managed_internal_ids: list[int] | None = None,
        managed_group_ids: list[int] | None = None,
    ) -> bool:
        """Check if user can edit/add/delete chunks for this document.

        Admin can edit any document. Owner can edit their own documents.
        CURATOR can edit docs of assigned users and managed group docs.
        Group documents (internal_group) can only be edited by admin/curator
        with managed scope, since there is no single owner.
        """
        if user_role == UserRole.ADMIN:
            return True

        if self.owner_id == user_id:
            return True

        if user_role == UserRole.CURATOR:
            return self.in_curator_scope(
                user_group_ids,
                managed_client_ids=managed_client_ids,
                managed_internal_ids=managed_internal_ids,
                managed_group_ids=managed_group_ids,
            )

        return False
