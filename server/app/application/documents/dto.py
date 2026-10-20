"""Document-related DTOs -- immutable data-transfer objects for document upload and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired, TypedDict

from domain.knowledge.enums import DocDomain, SourceType


if TYPE_CHECKING:
    from datetime import datetime

    from application.act_versioning.dto import VersioningPlan
    from application.documents.ports.processing import (
        ContentExtractorPort,
        MetricsCollectorPort,
        PDFQualityAssessorPort,
        TextQualityAssessorPort,
    )
    from application.ingestion import DocumentParserPort, DocumentSplitterPort
    from domain.ingestion.preview import PDFQualityReport
    from domain.ingestion.profiles.protocol import DocumentProfile
    from domain.ingestion.raw_document import RawDocument
    from domain.knowledge.document import Document
    from domain.knowledge.enums import DocumentVisibility


@dataclass(frozen=True)
class DocumentDTO:
    id: int
    filename: str
    visibility: str
    status: str
    source_path: str | None = None
    creation_date: datetime | None = None
    indexed_at: datetime | None = None
    error_message: str | None = None
    warning_message: str | None = None
    quality_score: float | None = None
    chunks: int | None = None
    chars: int | None = None
    storage_key: str | None = None
    replace_id: int | None = None
    owner_id: int | None = None
    group_id: int | None = None
    doc_domain: str = DocDomain.GENERAL.value
    source_type: str = SourceType.FILE.value
    has_manual_edits: bool = False
    version_group_id: int | None = None
    in_search_scope: bool = True
    outbox_pending: int = 0
    outbox_failed: int = 0
    outbox_failed_details: list[dict[str, object]] | None = None

    @classmethod
    def from_entity(cls, doc: Document, **overrides: object) -> DocumentDTO:
        if doc.id is None:
            msg = "Document must have an id to convert to DTO"
            raise ValueError(msg)

        defaults: dict[str, object] = {
            "source_path": doc.source_path,
            "creation_date": doc.creation_date,
            "indexed_at": doc.indexed_at,
            "error_message": doc.error_message,
            "warning_message": doc.warning_message,
            "quality_score": doc.quality_score,
            "chunks": doc.chunks,
            "chars": doc.chars,
            "owner_id": doc.owner_id,
            "group_id": doc.group_id,
            "doc_domain": doc.doc_domain,
            "source_type": doc.source_type,
            "has_manual_edits": doc.has_manual_edits,
            "version_group_id": doc.version_group_id,
        }
        defaults.update(overrides)
        return cls(
            id=doc.id,
            filename=doc.filename,
            visibility=doc.visibility,
            status=doc.status,
            **defaults,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ClientInfo:
    id: int
    email: str


@dataclass(frozen=True, kw_only=True)
class DocumentProcessingDependencies:
    document_parser: DocumentParserPort
    document_splitter: DocumentSplitterPort
    content_extractor: ContentExtractorPort
    pdf_quality_assessor: PDFQualityAssessorPort
    text_quality_assessor: TextQualityAssessorPort
    metrics: MetricsCollectorPort


@dataclass(frozen=True, kw_only=True)
class ProcessDocumentCommand:
    document_id: int
    storage_key: str
    original_filename: str
    visibility: DocumentVisibility
    owner_id: int | None
    group_id: int | None
    replace_id: int | None
    doc_domain: str | None = None


@dataclass(frozen=True)
class PreparedDocumentVersioning:
    plan: VersioningPlan
    profile: DocumentProfile


@dataclass(frozen=True, kw_only=True)
class PersistDocumentCommand:
    document_id: int
    original_filename: str
    raw_chunks: list[RawDocument]
    visibility: DocumentVisibility
    owner_id: int | None
    group_id: int | None
    doc_domain: str
    replace_id: int | None
    warning_message: str | None
    quality: PDFQualityReport | None
    storage_deletes: list[str]
    versioning: PreparedDocumentVersioning | None = None


class UploadOptions(TypedDict):
    user_id: int
    user_kind: str
    user_role: str
    client_id: NotRequired[int | None]
    rename_on_conflict: NotRequired[bool]
    doc_domain: NotRequired[str | None]
    replaces_document_id: NotRequired[int | None]
