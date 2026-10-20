"""Public feature exports match their implementations."""

import importlib

import pytest


@pytest.mark.parametrize(
    "public,implementation,name",
    [
        ("application.ingestion", "application.ingestion.service", "IngestionService"),
        ("application.ingestion", "application.ingestion.batch", "BatchIngestionWorkflow"),
        ("application.ingestion", "application.ingestion.single_file", "SingleFileIngestionWorkflow"),
        ("application.ingestion", "application.ingestion.registry", "IngestionRegistry"),
        ("application.ingestion", "application.ingestion.scope", "IngestionScope"),
        ("application.ingestion", "application.ingestion.sync", "DocumentSyncService"),
        ("application.ingestion", "application.ingestion.targets", "S3IngestionTargets"),
        ("application.ingestion", "application.ingestion.targets", "S3UploadService"),
        ("application.ingestion", "application.ingestion.loader", "S3DocumentLoader"),
        ("application.ingestion", "application.ingestion.ports.ingestion", "IngestionPort"),
        ("application.ingestion", "application.ingestion.ports.settings", "IngestionSettingsPort"),
        ("application.ingestion", "application.ingestion.ports.parser", "FileMeta"),
        ("application.ingestion", "application.ingestion.ports.parser", "SplitContext"),
        ("application.ingestion", "application.ingestion.ports.parser", "DocumentParserPort"),
        ("application.ingestion", "application.ingestion.ports.parser", "DocumentSplitterPort"),
        ("application.ingestion", "application.ingestion.dto", "IngestStatusResult"),
        ("application.ingestion", "application.ingestion.dto", "IngestRegistryItemDTO"),
        ("application.ingestion", "application.ingestion.dto", "IngestRegistryResult"),
        ("application.ingestion", "application.ingestion.dto", "RegistryRecord"),
        ("application.ingestion", "application.ingestion.dto", "SyncOneDocumentOptions"),
        ("application.ingestion", "application.ingestion.ports.upload", "UploadFile"),
    ],
)
def test_public_exports_share_canonical_objects(public: str, implementation: str, name: str) -> None:
    assert name in importlib.import_module(public).__all__
    assert getattr(importlib.import_module(public), name) is getattr(
        importlib.import_module(implementation), name
    )


def test_processing_details_are_not_public_exports() -> None:
    public = importlib.import_module("application.ingestion")
    assert "s3_file_hash" not in public.__all__
    assert not hasattr(public, "s3_file_hash")
