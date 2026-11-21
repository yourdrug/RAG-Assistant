"""Document-related DTOs -- immutable data-transfer objects for document upload and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired, TypedDict

from domain.knowledge.enums import DocDomain, SourceType


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from datetime import datetime

    from application.documents.ports.processing import (
        ContentExtractorPort,
        MetricsCollectorPort,
        PDFQualityAssessorPort,
        TextQualityAssessorPort,
    )
    from application.indexing.contextualization import ChunkContextualization
    from application.indexing.pack_hooks import PreparedPackDocument
    from application.ingestion import DocumentParserPort, DocumentSplitterPort
    from application.uow import UnitOfWork
    from domain.ingestion.preview import PDFQualityReport
    from domain.ingestion.raw_document import RawDocument
    from domain.knowledge.document import Document
    from domain.knowledge.enums import DocumentVisibility
    from domain.knowledge.revision import SourceVersion


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
    space_id: int | None = None
    doc_domain: str = DocDomain.GENERAL.value
    source_type: str = SourceType.FILE.value
    has_manual_edits: bool = False
    version_group_id: int | None = None
    in_search_scope: bool = True
    outbox_pending: int = 0
    outbox_failed: int = 0
    outbox_failed_details: list[dict[str, object]] | None = None
    # The sync channel that manages the document and its storage path; manual edits are refused.
    sync_channel_id: int | None = None
    sync_path: str | None = None

    @classmethod
    def from_entity(
        cls,
        doc: Document,
        *,
        storage_key: str | None = None,
        replace_id: int | None = None,
        in_search_scope: bool = True,
    ) -> DocumentDTO:
        if doc.id is None:
            msg = "Document must have an id to convert to DTO"
            raise ValueError(msg)

        return cls(
            id=doc.id,
            filename=doc.filename,
            visibility=doc.visibility,
            status=doc.status,
            source_path=doc.source_path,
            creation_date=doc.creation_date,
            indexed_at=doc.indexed_at,
            error_message=doc.error_message,
            warning_message=doc.warning_message,
            quality_score=doc.quality_score,
            chunks=doc.chunks,
            chars=doc.chars,
            owner_id=doc.owner_id,
            group_id=doc.group_id,
            space_id=doc.space_id,
            doc_domain=doc.doc_domain,
            source_type=doc.source_type,
            has_manual_edits=doc.has_manual_edits,
            version_group_id=doc.version_group_id,
            storage_key=storage_key,
            replace_id=replace_id,
            in_search_scope=in_search_scope,
            sync_channel_id=doc.sync.channel_id if doc.sync else None,
            sync_path=doc.sync.path if doc.sync else None,
        )


@dataclass(frozen=True)
class DocumentOverviewDTO:
    total: int
    chunks: int
    by_status: dict[str, int]
    failed: list[DocumentDTO]


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
    chunk_contextualization: ChunkContextualization | None = None


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
    # Re-chunking a stored document: its state before the run. A failure before the
    # new chunks are committed restores it, and manual chunk edits abort the commit.
    reindex_snapshot: Document | None = None
    skip_persisted: bool = False


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
    pack_document: PreparedPackDocument | None = None
    abort_on_manual_edits: bool = False
    # Runs first inside the persist transaction, before the document row lock (it locks
    # coarser rows first); raising aborts the commit (a sync run fences its writes).
    guard: PersistGuard | None = None
    # Source version of the content; recorded on the document when its chunks are published.
    source_version: SourceVersion | None = None


type PersistGuard = Callable[[UnitOfWork], Awaitable[None]]


@dataclass(frozen=True, kw_only=True)
class PreparedDocument:
    """Content ready to persist: prepared without any open transaction."""

    raw_chunks: list[RawDocument]
    doc_domain: str
    pack_document: PreparedPackDocument | None
    warnings: tuple[str, ...]
    quality: PDFQualityReport | None


class UploadOptions(TypedDict):
    user_id: int
    user_kind: str
    user_role: str
    client_id: NotRequired[int | None]
    rename_on_conflict: NotRequired[bool]
    doc_domain: NotRequired[str | None]
    replaces_document_id: NotRequired[int | None]
    # Knowledge space to fill; the only space is used when omitted.
    space_id: NotRequired[int | None]
