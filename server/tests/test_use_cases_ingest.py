"""
Tests for ingest use cases: RunIngestion, IngestSingleFile, GetIngestRegistry, IngestAppService.
All dependencies are mocked.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from application.services.ingest_service import IngestAppService
from application.use_cases.ingest.get_registry import GetIngestRegistry
from application.use_cases.ingest.ingest_single_file import IngestSingleFile
from application.use_cases.ingest.run_ingestion import RunIngestion

# ---------------------------------------------------------------------------
# RunIngestion
# ---------------------------------------------------------------------------


class TestRunIngestion:
    def setup_method(self):
        self.registry_repo = MagicMock()
        self.ingestion_service = MagicMock()
        self.use_case = RunIngestion(self.registry_repo, self.ingestion_service)

    def test_full_ingestion_default(self):
        self.use_case.execute(docs_dir="/data/docs")

        self.ingestion_service.run_full_ingestion.assert_called_once_with(
            "/data/docs", reset=False, prefix=None
        )

    def test_full_ingestion_with_reset(self):
        self.use_case.execute(docs_dir="/data/docs", reset=True)

        self.ingestion_service.run_full_ingestion.assert_called_once_with(
            "/data/docs", reset=True, prefix=None
        )

    def test_full_ingestion_with_prefix(self):
        self.use_case.execute(docs_dir="/data/docs", prefix="v2/")

        self.ingestion_service.run_full_ingestion.assert_called_once_with(
            "/data/docs", reset=False, prefix="v2/"
        )


# ---------------------------------------------------------------------------
# IngestSingleFile
# ---------------------------------------------------------------------------


class TestIngestSingleFile:
    def setup_method(self):
        self.registry_repo = MagicMock()
        self.ingestion_service = MagicMock()
        self.use_case = IngestSingleFile(self.registry_repo, self.ingestion_service)

    def test_single_file_without_force(self):
        self.use_case.execute(file_path="/data/doc.pdf", force=False)

        self.ingestion_service.force_reindex.assert_not_called()
        self.ingestion_service.run_single_file.assert_called_once_with("/data/doc.pdf")

    def test_single_file_with_force(self):
        self.use_case.execute(file_path="/data/doc.pdf", force=True)

        self.ingestion_service.force_reindex.assert_called_once_with("doc.pdf")
        self.ingestion_service.run_single_file.assert_called_once_with("/data/doc.pdf")

    def test_force_extracts_filename(self):
        self.use_case.execute(file_path="/some/deep/path/report.pdf", force=True)

        self.ingestion_service.force_reindex.assert_called_once_with("report.pdf")


# ---------------------------------------------------------------------------
# GetIngestRegistry
# ---------------------------------------------------------------------------


class TestGetIngestRegistry:
    def setup_method(self):
        self.registry_repo = MagicMock()
        self.use_case = GetIngestRegistry(self.registry_repo)

    def test_empty_registry(self):
        self.registry_repo.load.return_value = {}

        result = self.use_case.execute()

        assert result.total_files == 0
        assert result.total_chunks == 0
        assert result.files == []

    def test_registry_with_files(self):
        self.registry_repo.load.return_value = {
            "doc1.pdf": {"chunks": 5, "chars": 1000, "indexed_at": "2024-01-01", "source": "s3://b/doc1.pdf"},
            "doc2.pdf": {"chunks": 3, "chars": 500, "indexed_at": "2024-01-02", "source": "s3://b/doc2.pdf"},
        }

        result = self.use_case.execute()

        assert result.total_files == 2
        assert result.total_chunks == 8
        # Should be sorted by filename
        assert result.files[0].filename == "doc1.pdf"
        assert result.files[1].filename == "doc2.pdf"

    def test_registry_with_missing_metadata(self):
        self.registry_repo.load.return_value = {"doc.pdf": {}}

        result = self.use_case.execute()

        assert result.total_files == 1
        assert result.files[0].chunks == 0
        assert result.files[0].chars == 0
        assert result.files[0].indexed_at == ""
        assert result.files[0].source == ""

    def test_registry_sorted_alphabetically(self):
        self.registry_repo.load.return_value = {
            "z_doc.pdf": {"chunks": 1},
            "a_doc.pdf": {"chunks": 2},
            "m_doc.pdf": {"chunks": 3},
        }

        result = self.use_case.execute()

        filenames = [f.filename for f in result.files]
        assert filenames == ["a_doc.pdf", "m_doc.pdf", "z_doc.pdf"]


# ---------------------------------------------------------------------------
# IngestAppService
# ---------------------------------------------------------------------------


class TestIngestAppService:
    def setup_method(self):
        self.run_ingestion = MagicMock()
        self.ingest_single_file = MagicMock()
        self.get_registry = MagicMock()
        self.path_resolver = MagicMock()
        self.service = IngestAppService(
            self.run_ingestion,
            self.ingest_single_file,
            self.get_registry,
            self.path_resolver,
        )

    def test_run_full_default_mode(self):
        self.path_resolver.resolve_docs_dir.return_value = "/resolved/docs"

        result = self.service.run_full(docs_dir="docs_sample")

        assert result.status == "started"
        assert result.mode == "APPEND (new files only)"
        assert result.docs_dir == "/resolved/docs"

    def test_run_full_reset_mode(self):
        self.path_resolver.resolve_docs_dir.return_value = "/resolved/docs"

        result = self.service.run_full(docs_dir="docs_sample", reset=True)

        assert result.mode == "RESET + full reindex"

    def test_run_single(self):
        self.path_resolver.resolve_ingest_target.return_value = "/resolved/doc.pdf"

        result = self.service.run_single(file_path="doc.pdf")

        assert result.status == "started"
        assert result.file == "/resolved/doc.pdf"
        assert result.force is False

    def test_run_single_with_force(self):
        self.path_resolver.resolve_ingest_target.return_value = "/resolved/doc.pdf"

        result = self.service.run_single(file_path="doc.pdf", force=True)

        assert result.force is True

    def test_get_registry(self):
        expected = MagicMock()
        self.get_registry.execute.return_value = expected

        result = self.service.get_registry()

        assert result == expected

    def test_resolve_docs_dir(self):
        self.path_resolver.resolve_docs_dir.return_value = "/resolved"

        result = self.service.resolve_docs_dir("docs")

        assert result == "/resolved"
        self.path_resolver.resolve_docs_dir.assert_called_once_with("docs")

    def test_resolve_ingest_target(self):
        self.path_resolver.resolve_ingest_target.return_value = "/resolved/file.pdf"

        result = self.service.resolve_ingest_target("file.pdf")

        assert result == "/resolved/file.pdf"

    def test_force_reindex(self):
        self.service.force_reindex("doc.pdf")

        self.path_resolver.force_reindex.assert_called_once_with("doc.pdf")
