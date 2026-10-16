"""Compatibility of legacy exports with the canonical feature package."""

import importlib

import pytest


@pytest.mark.parametrize(
    "legacy,canonical,name",
    [
        ("application.services.ingest_service", "application.ingestion.service", "IngestAppService"),
        (
            "application.services.ingestion_orchestrator",
            "application.ingestion.orchestrator",
            "IngestionService",
        ),
        ("application.services.batch_ingestion", "application.ingestion.batch", "BatchIngestionWorkflow"),
        (
            "application.services.single_file_ingestion",
            "application.ingestion.single_file",
            "SingleFileIngestionWorkflow",
        ),
        ("application.services.ingestion_registry", "application.ingestion.registry", "s3_file_hash"),
        ("application.services.ingestion_registry", "application.ingestion.registry", "IngestionRegistry"),
        ("application.services.ingestion_scope", "application.ingestion.scope", "IngestionScope"),
        ("application.services.ingestion_sync", "application.ingestion.sync", "DocumentSyncService"),
        ("application.services.ingestion_targets", "application.ingestion.targets", "S3IngestionTargets"),
        ("application.services.ingestion_targets", "application.ingestion.targets", "S3UploadService"),
        ("application.services.document_loader", "application.ingestion.loader", "S3DocumentLoader"),
        (
            "application.services.sparse_index_builder",
            "application.ingestion.sparse_index",
            "SparseIndexBuilder",
        ),
        ("application.ports.ingestion_port", "application.ingestion.port", "IngestionPort"),
        ("application.ports.ingestion_settings", "application.ingestion.settings", "IngestionSettingsPort"),
        ("application.ports.document_parser", "application.ingestion.parser", "FileMeta"),
        ("application.ports.document_parser", "application.ingestion.parser", "SplitContext"),
        ("application.ports.document_parser", "application.ingestion.parser", "DocumentParserPort"),
        ("application.ports.document_parser", "application.ingestion.parser", "DocumentSplitterPort"),
        ("application.dto.ingest_dto", "application.ingestion.dto", "IngestStatusResult"),
        ("application.dto.ingest_dto", "application.ingestion.dto", "IngestRegistryItemDTO"),
        ("application.dto.ingest_dto", "application.ingestion.dto", "IngestRegistryResult"),
    ],
)
def test_legacy_exports_share_canonical_objects(legacy: str, canonical: str, name: str) -> None:
    assert getattr(importlib.import_module(legacy), name) is getattr(importlib.import_module(canonical), name)
