"""Public services, composition components and contracts; processing helpers stay in submodules."""

from application.documents.access import (
    DocumentAccessService,
)
from application.documents.access_port import (
    DocumentAccessPort,
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
    DocumentProcessorOptions,
    ProcessOptions,
    UploadOptions,
)
from application.documents.manual import (
    ManualDocumentService,
)
from application.documents.preview_cache import (
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
    "DocumentProcessor",
    "DocumentProcessorOptions",
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
    "ProcessOptions",
    "QualityService",
    "TextQualityAssessorPort",
    "UploadOptions",
]
