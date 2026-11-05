"""Tests for IngestService S3 key validation.

Tests resolve_ingest_target / resolve_docs_dir — protections for
POST /ingest and POST /ingest/file against invalid S3 keys.
"""

from pathlib import Path
import sys
from unittest.mock import MagicMock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest

from application.ingestion import (
    IngestionService,
    S3IngestionTargets,
    S3UploadService,
)


@pytest.fixture
def service() -> IngestionService:

    targets = S3IngestionTargets()
    return IngestionService(
        batch_workflow=MagicMock(),
        single_file_workflow=MagicMock(),
        registry=MagicMock(),
        targets=targets,
        uploads=S3UploadService(MagicMock(), targets),
    )


def test_valid_s3_key_resolves(service: IngestionService) -> None:
    resolved = service.resolve_ingest_target("docs/report.pdf")
    assert resolved == "docs/report.pdf"


def test_nested_s3_key_resolves(service: IngestionService) -> None:
    resolved = service.resolve_ingest_target("docs/subfolder/report.pdf")
    assert resolved == "docs/subfolder/report.pdf"


def test_s3_key_with_slash_prefix_is_rejected(service: IngestionService) -> None:
    with pytest.raises(ValueError, match="S3 key must not contain"):
        service.resolve_ingest_target("/docs/report.pdf")


def test_s3_key_with_dotdot_is_rejected(service: IngestionService) -> None:
    with pytest.raises(ValueError, match="S3 key must not contain"):
        service.resolve_ingest_target("docs/../etc/passwd")


def test_docs_dir_with_dotdot_is_rejected(service: IngestionService) -> None:
    with pytest.raises(ValueError, match="S3 key must not contain"):
        service.resolve_docs_dir("docs/../other/")


def test_docs_dir_with_slash_prefix_is_rejected(service: IngestionService) -> None:
    with pytest.raises(ValueError, match="S3 key must not contain"):
        service.resolve_docs_dir("/docs/")


def test_resolve_docs_dir_passes_prefix(service: IngestionService) -> None:
    resolved = service.resolve_docs_dir("my-prefix/")
    assert resolved == "my-prefix/"


def test_resolve_docs_dir_default_prefix(service: IngestionService) -> None:
    resolved = service.resolve_docs_dir("docs/")
    assert resolved == "docs/"
