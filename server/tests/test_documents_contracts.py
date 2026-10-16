"""Compatibility of legacy exports with the canonical feature package."""

import importlib

import pytest


@pytest.mark.parametrize(
    "legacy,canonical,name",
    [
        (
            "application.services.document_access_service",
            "application.documents.access",
            "DocumentAccessService",
        ),
        (
            "application.services.document_command_service",
            "application.documents.commands",
            "DocumentCommandService",
        ),
        (
            "application.services.document_conflict_resolver",
            "application.documents.conflicts",
            "resolve_upload_version_group",
        ),
        (
            "application.services.document_conflict_resolver",
            "application.documents.conflicts",
            "resolve_upload_conflict",
        ),
        (
            "application.services.document_conflict_resolver",
            "application.documents.conflicts",
            "resolve_processing_conflict",
        ),
        (
            "application.services.document_conflict_resolver",
            "application.documents.conflicts",
            "resolve_conflict",
        ),
        (
            "application.services.document_conflict_resolver",
            "application.documents.conflicts",
            "resolve_conflict_in_uow",
        ),
        (
            "application.services.document_conflict_resolver",
            "application.documents.conflicts",
            "apply_async_replacement",
        ),
        (
            "application.services.document_content",
            "application.documents.content",
            "DocumentContentProcessor",
        ),
        (
            "application.services.document_delete_command",
            "application.documents.delete",
            "DocumentDeleteCommand",
        ),
        (
            "application.services.document_deletion_jobs",
            "application.documents.deletion_jobs",
            "DocumentDeletionJobs",
        ),
        (
            "application.services.document_persistence",
            "application.documents.persistence",
            "persist_document_result",
        ),
        ("application.services.document_processor", "application.documents.processor", "DocumentProcessor"),
        ("application.services.document_quality", "application.documents.quality", "QualityOutcome"),
        ("application.services.document_quality", "application.documents.quality", "assess_document_quality"),
        (
            "application.services.document_query_service",
            "application.documents.queries",
            "DocumentQueryService",
        ),
        (
            "application.services.document_rename_command",
            "application.documents.rename",
            "DocumentRenameCommand",
        ),
        (
            "application.services.document_upload_command",
            "application.documents.upload",
            "DocumentUploadCommand",
        ),
        ("application.services.document_utils", "application.documents.utils", "generate_storage_key"),
        ("application.services.document_utils", "application.documents.utils", "resolve_unique_filename"),
        (
            "application.services.document_versioning",
            "application.documents.versioning",
            "prepare_processing_versioning",
        ),
        (
            "application.services.domain_classification",
            "application.documents.classification",
            "classify_processing_document",
        ),
        (
            "application.services.domain_classification",
            "application.documents.classification",
            "DomainClassification",
        ),
        (
            "application.services.domain_classification",
            "application.documents.classification",
            "classify_document_text",
        ),
        (
            "application.services.manual_document_service",
            "application.documents.manual",
            "ManualDocumentService",
        ),
        ("application.services.preview_cache", "application.documents.preview_cache", "DEFAULT_TTL_SECONDS"),
        ("application.services.preview_cache", "application.documents.preview_cache", "PREVIEW_PREFIX"),
        ("application.services.preview_cache", "application.documents.preview_cache", "PreviewCache"),
        ("application.ports.document_access", "application.documents.access_port", "DocumentAccessPort"),
        (
            "application.ports.document_preview",
            "application.documents.preview_port",
            "DocumentPreviewStrategy",
        ),
        (
            "application.ports.document_processing",
            "application.documents.processing_ports",
            "ContentExtractorPort",
        ),
        (
            "application.ports.document_processing",
            "application.documents.processing_ports",
            "PDFQualityAssessorPort",
        ),
        (
            "application.ports.document_processing",
            "application.documents.processing_ports",
            "TextQualityAssessorPort",
        ),
        (
            "application.ports.document_processing",
            "application.documents.processing_ports",
            "MetricsCollectorPort",
        ),
        (
            "application.ports.preview_strategy_factory",
            "application.documents.preview_factory",
            "PreviewStrategyFactoryPort",
        ),
        ("application.dto.document_dto", "application.documents.dto", "DocumentDTO"),
        ("application.dto.document_dto", "application.documents.dto", "ClientInfo"),
        (
            "application.dto.document_processing_context",
            "application.documents.processing_context",
            "empty_versioning_result",
        ),
        (
            "application.dto.document_processing_context",
            "application.documents.processing_context",
            "ProcessingContext",
        ),
    ],
)
def test_legacy_exports_share_canonical_objects(legacy: str, canonical: str, name: str) -> None:
    assert getattr(importlib.import_module(legacy), name) is getattr(importlib.import_module(canonical), name)
