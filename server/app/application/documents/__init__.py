"""Public services, composition components and contracts; processing helpers stay in submodules."""

from application.documents.access import (
    DocumentAccessService,
)
from application.documents.commands import (
    DocumentCommandService,
)
from application.documents.delete import (
    DocumentDeleteCommand,
)
from application.documents.deletion_jobs import (
    DocumentDeletionJobs,
)
from application.documents.diagnostics import (
    DocumentDiagnoseResult,
    PageDiagnostic,
    PDFDiagnosticService,
)
from application.documents.dto import (
    ClientInfo,
    DocumentDTO,
    DocumentProcessingDependencies,
    ProcessDocumentCommand,
    UploadOptions,
)
from application.documents.manual import (
    ManualDocumentService,
)
from application.documents.ports.access import (
    DocumentAccessPort,
)
from application.documents.ports.preview import (
    DocumentPreviewStrategy,
)
from application.documents.ports.preview_factory import (
    PreviewStrategyFactoryPort,
)
from application.documents.ports.processing import (
    ContentExtractorPort,
    MetricsCollectorPort,
    PDFQualityAssessorPort,
    TextQualityAssessorPort,
)
from application.documents.preview_cache import (
    PreviewCache,
)
from application.documents.processor import (
    DocumentProcessor,
)
from application.documents.quality_service import (
    QualityService,
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


__all__ = [
    "ClientInfo",
    "ContentExtractorPort",
    "DocumentAccessPort",
    "DocumentAccessService",
    "DocumentCommandService",
    "DocumentDTO",
    "DocumentDeleteCommand",
    "DocumentDeletionJobs",
    "DocumentDiagnoseResult",
    "DocumentPreviewStrategy",
    "DocumentProcessingDependencies",
    "DocumentProcessor",
    "DocumentQueryService",
    "DocumentRenameCommand",
    "DocumentUploadCommand",
    "ManualDocumentService",
    "MetricsCollectorPort",
    "PDFDiagnosticService",
    "PDFQualityAssessorPort",
    "PageDiagnostic",
    "PreviewCache",
    "PreviewStrategyFactoryPort",
    "ProcessDocumentCommand",
    "QualityService",
    "TextQualityAssessorPort",
    "UploadOptions",
]
