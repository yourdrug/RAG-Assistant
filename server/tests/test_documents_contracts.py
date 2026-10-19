"""Public feature exports match their implementations."""

import importlib

import pytest


@pytest.mark.parametrize(
    "public,implementation,name",
    [
        ("application.documents", "application.documents.access", "DocumentAccessService"),
        ("application.documents", "application.documents.commands", "DocumentCommandService"),
        ("application.documents", "application.documents.delete", "DocumentDeleteCommand"),
        ("application.documents", "application.documents.deletion_jobs", "DocumentDeletionJobs"),
        ("application.documents", "application.documents.processor", "DocumentProcessor"),
        ("application.documents", "application.documents.queries", "DocumentQueryService"),
        ("application.documents", "application.documents.rename", "DocumentRenameCommand"),
        ("application.documents", "application.documents.upload", "DocumentUploadCommand"),
        ("application.documents", "application.documents.manual", "ManualDocumentService"),
        ("application.documents", "application.documents.preview_cache", "PreviewCache"),
        ("application.documents", "application.documents.ports.access", "DocumentAccessPort"),
        ("application.documents", "application.documents.ports.preview", "DocumentPreviewStrategy"),
        ("application.documents", "application.documents.ports.processing", "ContentExtractorPort"),
        ("application.documents", "application.documents.ports.processing", "PDFQualityAssessorPort"),
        ("application.documents", "application.documents.ports.processing", "TextQualityAssessorPort"),
        ("application.documents", "application.documents.ports.processing", "MetricsCollectorPort"),
        (
            "application.documents",
            "application.documents.ports.preview_factory",
            "PreviewStrategyFactoryPort",
        ),
        ("application.documents", "application.documents.dto", "DocumentDTO"),
        ("application.documents", "application.documents.dto", "ClientInfo"),
        ("application.documents", "application.documents.diagnostics", "DocumentDiagnoseResult"),
        ("application.documents", "application.documents.diagnostics", "PageDiagnostic"),
        ("application.documents", "application.documents.diagnostics", "PDFDiagnosticService"),
        ("application.documents", "application.documents.dto", "DocumentProcessingDependencies"),
        ("application.documents", "application.documents.dto", "ProcessDocumentCommand"),
        ("application.documents", "application.documents.dto", "UploadOptions"),
        ("application.documents", "application.documents.quality_service", "QualityService"),
    ],
)
def test_public_exports_share_canonical_objects(public: str, implementation: str, name: str) -> None:
    assert name in importlib.import_module(public).__all__
    assert getattr(importlib.import_module(public), name) is getattr(
        importlib.import_module(implementation), name
    )


@pytest.mark.parametrize(
    "name",
    [
        "DEFAULT_TTL_SECONDS",
        "DocumentContentProcessor",
        "DomainClassification",
        "PREVIEW_PREFIX",
        "PersistDocumentCommand",
        "QualityOutcome",
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
    ],
)
def test_processing_details_are_not_public_exports(name: str) -> None:
    public = importlib.import_module("application.documents")
    assert name not in public.__all__
    assert not hasattr(public, name)
