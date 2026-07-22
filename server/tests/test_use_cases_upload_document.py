"""
Tests for UploadDocument use case.
Async, with mocked repos and services.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest
from application.use_cases.document.upload_document import UploadDocument
from domain.entities.document import Document
from domain.exceptions import BusinessRuleViolation, ValidationError
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.visibility import DocumentVisibility


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_repos(existing_doc=None):
    doc_repo = MagicMock()
    group_repo = MagicMock()
    document_processor = MagicMock()
    file_storage = MagicMock()
    file_storage.supported_extensions = [".pdf", ".docx", ".txt"]

    if existing_doc:
        doc_repo.find_active_slot.return_value = existing_doc
    else:
        doc_repo.find_active_slot.return_value = None

    saved_doc = Document(
        id=42,
        filename="doc.pdf",
        visibility=DocumentVisibility.INTERNAL_PRIVATE,
        owner_id=1,
        status=DocumentStatus.PENDING,
    )
    doc_repo.save.return_value = saved_doc
    doc_repo.get_by_id.return_value = saved_doc

    return doc_repo, group_repo, document_processor, file_storage


# ---------------------------------------------------------------------------
# UploadDocument
# ---------------------------------------------------------------------------


class TestUploadDocument:
    @pytest.mark.asyncio
    async def test_successful_upload(self):
        doc_repo, group_repo, processor, storage = _mock_repos()
        group_repo.get_user_group_ids.return_value = [10]

        use_case = UploadDocument(doc_repo, group_repo, processor, storage)

        result = await use_case.execute(
            filename="report.pdf",
            file_data=b"content",
            visibility="internal_private",
            group_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
        )

        assert result.id == 42
        doc_repo.save.assert_called_once()
        storage.upload_file.assert_called_once()
        processor.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsupported_extension_raises(self):
        doc_repo, group_repo, processor, storage = _mock_repos()
        group_repo.get_user_group_ids.return_value = []
        storage.supported_extensions = [".pdf"]

        use_case = UploadDocument(doc_repo, group_repo, processor, storage)

        with pytest.raises(ValidationError) as exc_info:
            await use_case.execute(
                filename="image.png",
                file_data=b"data",
                visibility="internal_private",
                group_id=None,
                user_id=1,
                user_kind="internal",
                user_role="user",
            )
        assert "Unsupported file format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_replacing_existing_done_document(self):
        existing = Document(
            id=10,
            filename="doc.pdf",
            status=DocumentStatus.DONE,
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
            owner_id=1,
        )
        doc_repo, group_repo, processor, storage = _mock_repos(existing_doc=existing)
        group_repo.get_user_group_ids.return_value = []

        use_case = UploadDocument(doc_repo, group_repo, processor, storage)

        await use_case.execute(
            filename="doc.pdf",
            file_data=b"new content",
            visibility="internal_private",
            group_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
        )

        # Should pass replace_id to processor
        call_kwargs = processor.process.call_args.kwargs
        assert call_kwargs["replace_id"] == 10

    @pytest.mark.asyncio
    async def test_rejects_document_being_processed(self):
        existing = Document(
            id=10,
            filename="doc.pdf",
            status=DocumentStatus.PROCESSING,
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
            owner_id=1,
        )
        doc_repo, group_repo, processor, storage = _mock_repos(existing_doc=existing)
        group_repo.get_user_group_ids.return_value = []

        use_case = UploadDocument(doc_repo, group_repo, processor, storage)

        with pytest.raises(BusinessRuleViolation) as exc_info:
            await use_case.execute(
                filename="doc.pdf",
                file_data=b"data",
                visibility="internal_private",
                group_id=None,
                user_id=1,
                user_kind="internal",
                user_role="user",
            )
        assert "already being processed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rejects_pending_document(self):
        existing = Document(
            id=10,
            filename="doc.pdf",
            status=DocumentStatus.PENDING,
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
            owner_id=1,
        )
        doc_repo, group_repo, processor, storage = _mock_repos(existing_doc=existing)
        group_repo.get_user_group_ids.return_value = []

        use_case = UploadDocument(doc_repo, group_repo, processor, storage)

        with pytest.raises(BusinessRuleViolation):
            await use_case.execute(
                filename="doc.pdf",
                file_data=b"data",
                visibility="internal_private",
                group_id=None,
                user_id=1,
                user_kind="internal",
                user_role="user",
            )

    @pytest.mark.asyncio
    async def test_storage_key_for_owner(self):
        doc_repo, group_repo, processor, storage = _mock_repos()
        group_repo.get_user_group_ids.return_value = []

        use_case = UploadDocument(doc_repo, group_repo, processor, storage)

        await use_case.execute(
            filename="report.pdf",
            file_data=b"data",
            visibility="internal_private",
            group_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
        )

        call_args = storage.upload_file.call_args
        key = call_args[0][0]
        assert key.startswith("uploads/users/1/")

    @pytest.mark.asyncio
    async def test_storage_key_for_public(self):
        doc_repo, group_repo, processor, storage = _mock_repos()
        group_repo.get_user_group_ids.return_value = []

        use_case = UploadDocument(doc_repo, group_repo, processor, storage)

        await use_case.execute(
            filename="report.pdf",
            file_data=b"data",
            visibility="internal_public",
            group_id=None,
            user_id=1,
            user_kind="internal",
            user_role="admin",
        )

        call_args = storage.upload_file.call_args
        key = call_args[0][0]
        assert key.startswith("uploads/public/")

    @pytest.mark.asyncio
    async def test_storage_key_for_group(self):
        doc_repo, group_repo, processor, storage = _mock_repos()
        group_repo.get_user_group_ids.return_value = [10]

        use_case = UploadDocument(doc_repo, group_repo, processor, storage)

        await use_case.execute(
            filename="report.pdf",
            file_data=b"data",
            visibility="internal_group",
            group_id=10,
            user_id=1,
            user_kind="internal",
            user_role="user",
        )

        call_args = storage.upload_file.call_args
        key = call_args[0][0]
        assert key.startswith("uploads/groups/10/")

    @pytest.mark.asyncio
    async def test_client_user_upload(self):
        doc_repo, group_repo, processor, storage = _mock_repos()
        group_repo.get_user_group_ids.return_value = []

        use_case = UploadDocument(doc_repo, group_repo, processor, storage)

        result = await use_case.execute(
            filename="client_doc.pdf",
            file_data=b"data",
            visibility="client_private",
            group_id=None,
            user_id=50,
            user_kind="client",
            user_role="user",
        )

        assert result.id == 42

    @pytest.mark.asyncio
    async def test_invalid_visibility_raises(self):
        doc_repo, group_repo, processor, storage = _mock_repos()

        use_case = UploadDocument(doc_repo, group_repo, processor, storage)

        with pytest.raises(ValidationError):
            await use_case.execute(
                filename="doc.pdf",
                file_data=b"data",
                visibility="nonexistent",
                group_id=None,
                user_id=1,
                user_kind="internal",
                user_role="user",
            )

    @pytest.mark.asyncio
    async def test_no_existing_slot_sets_replace_id_none(self):
        doc_repo, group_repo, processor, storage = _mock_repos(existing_doc=None)
        group_repo.get_user_group_ids.return_value = []

        use_case = UploadDocument(doc_repo, group_repo, processor, storage)

        await use_case.execute(
            filename="new.pdf",
            file_data=b"data",
            visibility="internal_private",
            group_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
        )

        call_kwargs = processor.process.call_args.kwargs
        assert call_kwargs["replace_id"] is None

    @pytest.mark.asyncio
    async def test_failed_existing_allows_reupload(self):
        existing = Document(
            id=10,
            filename="doc.pdf",
            status=DocumentStatus.FAILED,
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
            owner_id=1,
        )
        doc_repo, group_repo, processor, storage = _mock_repos(existing_doc=existing)
        group_repo.get_user_group_ids.return_value = []

        use_case = UploadDocument(doc_repo, group_repo, processor, storage)

        # Should not raise — failed documents can be re-uploaded
        await use_case.execute(
            filename="doc.pdf",
            file_data=b"data",
            visibility="internal_private",
            group_id=None,
            user_id=1,
            user_kind="internal",
            user_role="user",
        )

        call_kwargs = processor.process.call_args.kwargs
        assert call_kwargs["replace_id"] is None  # Failed doc doesn't get replaced


# ---------------------------------------------------------------------------
# _storage_key (static method)
# ---------------------------------------------------------------------------


class TestStorageKey:
    def test_owner_key(self):
        key = UploadDocument._storage_key(owner_id=1, group_id=None, document_id=42, filename="doc.pdf")
        assert key == "uploads/users/1/42_doc.pdf"

    def test_group_key(self):
        key = UploadDocument._storage_key(owner_id=None, group_id=10, document_id=42, filename="doc.pdf")
        assert key == "uploads/groups/10/42_doc.pdf"

    def test_public_key(self):
        key = UploadDocument._storage_key(owner_id=None, group_id=None, document_id=42, filename="doc.pdf")
        assert key == "uploads/public/42_doc.pdf"

    def test_strips_path_from_filename(self):
        key = UploadDocument._storage_key(owner_id=1, group_id=None, document_id=1, filename="/deep/path/doc.pdf")
        assert key == "uploads/users/1/1_doc.pdf"
