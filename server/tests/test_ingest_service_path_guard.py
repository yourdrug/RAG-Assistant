"""Tests for IngestService S3 key validation.

Tests resolve_ingest_target / resolve_docs_dir — protections for
POST /ingest and POST /ingest/file against invalid S3 keys.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest  # noqa: E402
from infrastructure.services.ingestion_service import IngestionService  # noqa: E402


@pytest.fixture
def service():
    from unittest.mock import MagicMock

    return IngestionService(
        vector_store_repo=MagicMock(),
        file_storage=MagicMock(),
    )


def test_valid_s3_key_resolves(service):
    resolved = service.resolve_ingest_target("docs/report.pdf")
    assert resolved == "docs/report.pdf"


def test_nested_s3_key_resolves(service):
    resolved = service.resolve_ingest_target("docs/subfolder/report.pdf")
    assert resolved == "docs/subfolder/report.pdf"


def test_s3_key_with_slash_prefix_is_rejected(service):
    with pytest.raises(ValueError):
        service.resolve_ingest_target("/docs/report.pdf")


def test_s3_key_with_dotdot_is_rejected(service):
    with pytest.raises(ValueError):
        service.resolve_ingest_target("docs/../etc/passwd")


def test_resolve_docs_dir_passes_prefix(service):
    resolved = service.resolve_docs_dir("my-prefix/")
    assert resolved == "my-prefix/"


def test_resolve_docs_dir_default_prefix(service):
    resolved = service.resolve_docs_dir("docs/")
    assert resolved == "docs/"
