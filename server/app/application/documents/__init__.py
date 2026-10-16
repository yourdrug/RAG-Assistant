"""Public documents operations, results and contracts."""

from application.documents.access import (
    DocumentAccessService,
)
from application.documents.access_port import (
    DocumentAccessPort,
)
from application.documents.classification import (
    DomainClassification,
    classify_document_text,
    classify_processing_document,
)
from application.documents.commands import (
    DocumentCommandService,
)
from application.documents.conflicts import (
    apply_async_replacement,
    resolve_conflict,
    resolve_conflict_in_uow,
    resolve_processing_conflict,
    resolve_upload_conflict,
    resolve_upload_version_group,
)
from application.documents.content import (
    DocumentContentProcessor,
)
from application.documents.delete import (
    DocumentDeleteCommand,
)
from application.documents.deletion_jobs import (
    DocumentDeletionJobs,
)
from application.documents.dto import (
    ClientInfo,
    DocumentDTO,
)
from application.documents.manual import (
    ManualDocumentService,
)
from application.documents.persistence import (
    persist_document_result,
)
from application.documents.preview_cache import (
    DEFAULT_TTL_SECONDS,
    PREVIEW_PREFIX,
    PreviewCache,
)
from application.documents.preview_factory import (
    PreviewStrategyFactoryPort,
)
from application.documents.preview_port import (
    DocumentPreviewStrategy,
)
from application.documents.processing_ports import (
    ContentExtractorPort,
    MetricsCollectorPort,
    PDFQualityAssessorPort,
    TextQualityAssessorPort,
)
from application.documents.processor import (
    DocumentProcessor,
)
from application.documents.quality import (
    QualityOutcome,
    assess_document_quality,
)
from application.documents.queries import (
    DocumentQueryService,
)
from application.documents.rename import (
    DocumentRenameCommand,
)
from application.documents.upload import (
    DocumentUploadCommand,
)
from application.documents.utils import (
    generate_storage_key,
    resolve_unique_filename,
)
from application.documents.versioning import (
    prepare_processing_versioning,
)


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "PREVIEW_PREFIX",
    "ClientInfo",
    "ContentExtractorPort",
    "DocumentAccessPort",
    "DocumentAccessService",
    "DocumentCommandService",
    "DocumentContentProcessor",
    "DocumentDTO",
    "DocumentDeleteCommand",
    "DocumentDeletionJobs",
    "DocumentPreviewStrategy",
    "DocumentProcessor",
    "DocumentProcessorOptions",
    "DocumentQueryService",
    "DocumentRenameCommand",
    "DocumentUploadCommand",
    "DomainClassification",
    "ManualDocumentService",
    "MetricsCollectorPort",
    "PDFQualityAssessorPort",
    "PersistDocumentResultOptions",
    "PreviewCache",
    "PreviewStrategyFactoryPort",
    "ProcessOptions",
    "QualityOutcome",
    "TextQualityAssessorPort",
    "UploadOptions",
    "apply_async_replacement",
    "assess_document_quality",
    "classify_document_text",
    "classify_processing_document",
    "generate_storage_key",
    "persist_document_result",
    "prepare_processing_versioning",
    "resolve_conflict",
    "resolve_conflict_in_uow",
    "resolve_processing_conflict",
    "resolve_unique_filename",
    "resolve_upload_conflict",
    "resolve_upload_version_group",
]

from application.documents.dto import (
    DocumentProcessorOptions,
    PersistDocumentResultOptions,
    ProcessOptions,
    UploadOptions,
)
