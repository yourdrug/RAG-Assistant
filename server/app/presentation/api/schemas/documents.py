"""Document schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.source_type import SourceType
from domain.value_objects.visibility import DocumentVisibility


class UploadStatusResponse(BaseModel):
    status: str
    document_id: int
    filename: str


class DocumentRenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(..., min_length=1, max_length=255)


class DocumentResponse(BaseModel):
    id: int
    filename: str
    source_path: str
    visibility: str
    owner_id: int | None
    group_id: int | None
    status: str
    doc_domain: str = DocDomain.GENERAL.value
    error_message: str | None
    warning_message: str | None
    quality_score: float | None = None
    chunks: int | None
    chars: int | None
    creation_date: datetime | None
    indexed_at: datetime | None
    source_type: str = SourceType.FILE.value
    has_manual_edits: bool = False
    outbox_pending: int = 0
    outbox_failed: int = 0
    outbox_failed_details: list[dict[str, object]] | None = None


class UploadResponse(BaseModel):
    files: list[str]


class ManualDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=255)
    visibility: DocumentVisibility
    group_id: int | None = None


class DeleteDocumentResponse(BaseModel):
    """Response for DELETE /documents/{id}."""

    status: str
    document_id: int
