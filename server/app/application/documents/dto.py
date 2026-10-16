"""Document-related DTOs -- immutable data-transfer objects for document upload and metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict


if TYPE_CHECKING:
    from application.act_versioning import ActVersioningService, VersioningPlan
    from application.documents.processing_ports import (
        ContentExtractorPort,
        MetricsCollectorPort,
        PDFQualityAssessorPort,
        TextQualityAssessorPort,
    )
    from application.ingestion import DocumentSplitterPort
    from domain.domain_profile.protocol import DomainProfile
    from domain.domain_profile.registry import DomainProfileRegistry
    from domain.domain_profile.settings_port import DomainSettingsPort
    from domain.value_objects.pdf_quality_report import PDFQualityReport

if TYPE_CHECKING:
    from application.act_versioning import VersioningResult
    from domain.entities.raw_document import RawDocument


from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.source_type import SourceType


if TYPE_CHECKING:
    from datetime import datetime

    from domain.entities.document import Document


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


class DocumentProcessorOptions(TypedDict):
    document_splitter: DocumentSplitterPort
    content_extractor: ContentExtractorPort
    pdf_quality_assessor: PDFQualityAssessorPort
    text_quality_assessor: TextQualityAssessorPort
    metrics: MetricsCollectorPort
    domain_marker_threshold: NotRequired[float]
    domain_registry: NotRequired[DomainProfileRegistry | None]
    domain_settings: NotRequired[DomainSettingsPort | None]
    act_versioning_service: NotRequired[ActVersioningService | None]


class ProcessOptions(TypedDict):
    owner_id: int | None
    group_id: int | None
    replace_id: int | None
    doc_domain: NotRequired[str | None]


class UploadOptions(TypedDict):
    user_id: int
    user_kind: str
    user_role: str
    client_id: NotRequired[int | None]
    rename_on_conflict: NotRequired[bool]
    doc_domain: NotRequired[str | None]
    replaces_document_id: NotRequired[int | None]


class PersistDocumentResultOptions(TypedDict):
    document_id: int
    original_filename: str
    raw_chunks: list[RawDocument]
    visibility: str
    owner_id: int | None
    group_id: int | None
    doc_domain: str
    replace_id: int | None
    warning_message: str | None
    quality: PDFQualityReport | None
    storage_deletes: list[str]
    domain_registry: DomainProfileRegistry | None
    versioning: VersioningResult
    versioning_plan: NotRequired[VersioningPlan | None]
    versioning_profile: NotRequired[DomainProfile | None]
    act_versioning_service: NotRequired[ActVersioningService | None]
