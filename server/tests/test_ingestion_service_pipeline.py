"""Characterization tests for services/ingestion_service.py main pipeline.

Locks down the orchestration logic (run_full_ingestion, _sync_documents_to_db,
_handle_s3_file, _load_documents, _parse_file) BEFORE moving the file
from infrastructure/ to application/.
"""

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from domain.value_objects.visibility import DocumentVisibility  # noqa: E402
from application.services.ingestion_orchestrator import (  # noqa: E402
    IngestionService,
    _s3_file_hash,
    _s3_source_key,
    _tag_chunks,
    _tag_domain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_pdf_settings():
    """Save and restore pdf.settings to prevent test-ordering pollution."""
    from config import settings as real_settings

    original_ocr_enabled = real_settings.ocr_enabled
    original_ocr_min_chars = real_settings.ocr_min_chars
    yield
    real_settings.ocr_enabled = original_ocr_enabled
    real_settings.ocr_min_chars = original_ocr_min_chars


def _make_file_item(**overrides) -> SimpleNamespace:
    defaults = {
        "key": "docs/test.pdf",
        "filename": "test.pdf",
        "extension": ".pdf",
        "size_bytes": 1024,
        "last_modified": datetime(2024, 1, 1),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_service(**overrides):
    mock_vector_store = MagicMock()
    mock_vector_store.ensure_collection = AsyncMock()
    mock_file_storage = MagicMock()
    mock_file_storage.supported_extensions = (".pdf", ".docx", ".txt", ".md")
    mock_file_storage.list_files = MagicMock(return_value=[])
    mock_file_storage.get_file_info = MagicMock(return_value=None)
    mock_file_storage.download_to_temp = AsyncMock(return_value=Path("/tmp/test.pdf"))

    defaults = {
        "vector_store_repo": mock_vector_store,
        "file_storage": mock_file_storage,
        "uow_factory": None,
        "domain_registry": None,
        "domain_settings": None,
        "act_versioning_service": None,
    }
    defaults.update(overrides)
    return IngestionService(**defaults)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestTagChunks:
    def test_tags_visibility(self):
        chunk = Document(page_content="text", metadata={})
        _tag_chunks([chunk], visibility=DocumentVisibility.INTERNAL_PUBLIC)
        assert chunk.metadata["visibility"] == "internal_public"

    def test_tags_owner_and_group(self):
        chunk = Document(page_content="text", metadata={})
        _tag_chunks([chunk], owner_id=42, group_id=7)
        assert chunk.metadata["owner_id"] == 42
        assert chunk.metadata["group_id"] == 7

    def test_tags_client_id(self):
        chunk = Document(page_content="text", metadata={})
        _tag_chunks([chunk], client_id=99)
        assert chunk.metadata["client_id"] == 99


class TestTagDomain:
    def test_sets_domain(self):
        chunk = Document(page_content="text", metadata={})
        _tag_domain([chunk], "legal")
        assert chunk.metadata["doc_domain"] == "legal"


class TestS3Helpers:
    def test_s3_file_hash(self):
        item = _make_file_item(size_bytes=1024, last_modified=datetime(2024, 6, 15))
        result = _s3_file_hash(item)
        assert "1024" in result

    def test_s3_source_key(self):
        item = _make_file_item(key="docs/test.pdf")
        with patch("application.services.ingestion_orchestrator.settings") as s:
            s.s3_bucket = "my-bucket"
            result = _s3_source_key(item)
            assert result == "s3://my-bucket/docs/test.pdf"


# ---------------------------------------------------------------------------
# _validate_s3_key
# ---------------------------------------------------------------------------


class TestValidateS3Key:
    def test_valid_key(self):
        IngestionService._validate_s3_key("docs/file.pdf")

    def test_leading_slash_raises(self):
        with pytest.raises(ValueError):
            IngestionService._validate_s3_key("/docs/file.pdf")

    def test_dotdot_raises(self):
        with pytest.raises(ValueError):
            IngestionService._validate_s3_key("docs/../file.pdf")


# ---------------------------------------------------------------------------
# _classify_text_domain
# ---------------------------------------------------------------------------


class TestClassifyTextDomain:
    def test_explicit_domain_returned(self):
        svc = _make_service()
        result = svc._classify_text_domain("text", "legal")
        assert result == "legal"

    def test_auto_calls_classify_domain(self):
        svc = _make_service()
        with patch("application.services.ingestion_orchestrator.classify_domain", return_value="general"):
            result = svc._classify_text_domain("some text", "auto")
            assert result == "general"


# ---------------------------------------------------------------------------
# _get_profile
# ---------------------------------------------------------------------------


class TestGetProfile:
    def test_no_registry(self):
        svc = _make_service(domain_registry=None)
        assert svc._get_profile("legal") is None

    def test_existing_profile(self):
        mock_registry = MagicMock()
        mock_profile = MagicMock()
        mock_registry.get.return_value = mock_profile
        svc = _make_service(domain_registry=mock_registry)
        assert svc._get_profile("legal") is mock_profile

    def test_missing_profile(self):
        mock_registry = MagicMock()
        mock_registry.get.side_effect = KeyError("not found")
        svc = _make_service(domain_registry=mock_registry)
        assert svc._get_profile("nonexistent") is None


# ---------------------------------------------------------------------------
# _parse_file
# ---------------------------------------------------------------------------


class TestParseFile:
    def test_pdf_parsing(self, tmp_path):
        import fitz

        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello world content for testing.")
        doc.save(str(pdf_path))
        doc.close()

        svc = _make_service()
        file_item = _make_file_item(extension=".pdf")
        with patch("infrastructure.ml.ingestion.pdf.settings") as mock_settings:
            mock_settings.ocr_enabled = False
            mock_settings.ocr_min_chars = 50
            try:
                result = svc._parse_file(file_item, pdf_path)
            except Exception as e:
                pytest.fail(f"_parse_file raised: {e}")
        assert result is not None, f"parse_pdf returned None (ocr_enabled={mock_settings.ocr_enabled})"
        assert len(result) >= 1

    def test_unsupported_extension(self):
        svc = _make_service()
        file_item = _make_file_item(extension=".xyz")
        result = svc._parse_file(file_item, Path("/tmp/x.xyz"))
        assert result is None

    def test_txt_parsing(self, tmp_path):
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("This is a test document with enough content to pass validation.")

        svc = _make_service()
        file_item = _make_file_item(filename="test.txt", extension=".txt")
        result = svc._parse_file(file_item, txt_path)
        assert result is not None
        assert len(result) == 1

    def test_base_metadata_added(self, tmp_path):
        import fitz

        pdf_path = tmp_path / "meta.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Metadata test content.")
        doc.save(str(pdf_path))
        doc.close()

        svc = _make_service()
        file_item = _make_file_item()
        with patch("infrastructure.ml.ingestion.pdf.settings") as mock_settings:
            mock_settings.ocr_enabled = False
            mock_settings.ocr_min_chars = 10
            result = svc._parse_file(file_item, pdf_path)
        assert result is not None
        meta = result[0].metadata
        assert "source" in meta
        assert "filename" in meta
        assert meta["filename"] == "test.pdf"


# ---------------------------------------------------------------------------
# _build_bm25_index
# ---------------------------------------------------------------------------


class TestBuildBm25Index:
    @pytest.mark.asyncio
    async def test_skips_when_hybrid_disabled(self):
        svc = _make_service()
        with patch("application.services.ingestion_orchestrator.settings") as s:
            s.hybrid_enabled = False
            await svc._build_bm25_index([], reset=False)
            # No error, just returns

    @pytest.mark.asyncio
    async def test_reset_builds_from_scratch(self):
        svc = _make_service()
        with (
            patch("application.services.ingestion_orchestrator.settings") as s,
            patch("application.services.ingestion_orchestrator.BM25Index") as mock_idx,
            patch(
                "application.services.ingestion_orchestrator.save_bm25_index_to_s3", new_callable=AsyncMock
            ),
            patch(
                "application.services.ingestion_orchestrator.publish_bm25_invalidation",
                new_callable=AsyncMock,
            ),
        ):
            s.hybrid_enabled = True
            chunks = [Document(page_content="hello world"), Document(page_content="foo bar")]
            await svc._build_bm25_index(chunks, reset=True)
            mock_idx.assert_called_once_with(
                ["hello world", "foo bar"],
                doc_visibility=[None, None],
                doc_owner_id=[None, None],
                doc_group_id=[None, None],
            )


# ---------------------------------------------------------------------------
# _registry operations (no uow_factory)
# ---------------------------------------------------------------------------


class TestRegistryOperations:
    @pytest.mark.asyncio
    async def test_upsert_no_uow(self):
        svc = _make_service(uow_factory=None)
        await svc._registry_upsert("file.pdf", "hash", "src", 10, 500)

    @pytest.mark.asyncio
    async def test_is_indexed_no_uow(self):
        svc = _make_service(uow_factory=None)
        result = await svc._registry_is_indexed("file.pdf", "hash")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_all_no_uow(self):
        svc = _make_service(uow_factory=None)
        result = await svc._registry_list_all()
        assert result == {}

    @pytest.mark.asyncio
    async def test_delete_no_uow(self):
        svc = _make_service(uow_factory=None)
        await svc._registry_delete("file.pdf")


# ---------------------------------------------------------------------------
# _log_ingest_config
# ---------------------------------------------------------------------------


class TestLogIngestConfig:
    def test_no_error(self):
        IngestionService._log_ingest_config(reset=True, docs_dir="my-docs/")
        IngestionService._log_ingest_config(reset=False, docs_dir=None)


# ---------------------------------------------------------------------------
# resolve_ingest_target / resolve_docs_dir
# ---------------------------------------------------------------------------


class TestResolveTargets:
    def test_resolve_ingest_target(self):
        svc = _make_service()
        assert svc.resolve_ingest_target("docs/file.pdf") == "docs/file.pdf"

    def test_resolve_docs_dir(self):
        svc = _make_service()
        assert svc.resolve_docs_dir("my-prefix/") == "my-prefix/"


# ---------------------------------------------------------------------------
# get_registry / force_reindex
# ---------------------------------------------------------------------------


class TestGetRegistryForceReindex:
    @pytest.mark.asyncio
    async def test_get_registry_no_uow(self):
        svc = _make_service(uow_factory=None)
        result = await svc.get_registry()
        assert result == {}

    @pytest.mark.asyncio
    async def test_force_reindex_no_uow(self):
        svc = _make_service(uow_factory=None)
        await svc.force_reindex("file.pdf")


# ---------------------------------------------------------------------------
# run_full_ingestion — no documents path
# ---------------------------------------------------------------------------


class TestRunFullIngestion:
    @pytest.mark.asyncio
    async def test_no_documents_returns_early(self):
        svc = _make_service()
        svc._file_storage.list_files = MagicMock(return_value=[])
        # Should not raise
        await svc.run_full_ingestion(docs_dir="empty/", reset=False)

    @pytest.mark.asyncio
    async def test_ensure_collection_called(self):
        svc = _make_service()
        svc._file_storage.list_files = MagicMock(return_value=[])
        await svc.run_full_ingestion(docs_dir="empty/", reset=True)
        svc._vector_store.ensure_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_deletes_internal_documents(self):
        svc = _make_service()
        svc._file_storage.list_files = MagicMock(return_value=[])
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_session.close = AsyncMock()

        mock_uow = MagicMock()
        mock_uow.documents.delete_internal_documents = AsyncMock(return_value=5)
        mock_uow._session = mock_session

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_uow)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        svc._uow_factory = MagicMock()
        svc._uow_factory.create = MagicMock(return_value=mock_ctx)

        await svc.run_full_ingestion(docs_dir="empty/", reset=True)
        mock_uow.documents.delete_internal_documents.assert_called_once()
