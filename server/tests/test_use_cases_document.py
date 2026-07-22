"""
Tests for document use cases: DeleteDocument, GetDocument, ListDocuments.
UploadDocument is async and has complex deps — tested separately.
All dependencies are mocked.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest
from application.use_cases.document.delete_document import DeleteDocument
from application.use_cases.document.get_document import GetDocument
from application.use_cases.document.list_documents import ListDocuments
from domain.entities.document import Document
from domain.exceptions import BusinessRuleViolation, EntityNotFound
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.visibility import DocumentVisibility

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(
    id=1,
    filename="doc.pdf",
    visibility=DocumentVisibility.INTERNAL_PUBLIC,
    owner_id=None,
    group_id=None,
    status=DocumentStatus.DONE,
    source_path="uploads/doc.pdf",
):
    return Document(
        id=id,
        filename=filename,
        visibility=visibility,
        owner_id=owner_id,
        group_id=group_id,
        status=status,
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# DeleteDocument
# ---------------------------------------------------------------------------


class TestDeleteDocument:
    def setup_method(self):
        self.doc_repo = MagicMock()
        self.vector_store_repo = MagicMock()
        self.file_storage = MagicMock()
        self.use_case = DeleteDocument(self.doc_repo, self.vector_store_repo, self.file_storage)

    def test_owner_can_delete_own_document(self):
        doc = _make_doc(id=10, owner_id=5)
        self.doc_repo.get_by_id.return_value = doc

        self.use_case.execute(document_id=10, user_id=5, user_role="user")

        self.vector_store_repo.delete_by_document_id.assert_called_once_with(10)
        self.file_storage.delete_file.assert_called_once_with("uploads/doc.pdf")
        self.doc_repo.delete.assert_called_once_with(10)

    def test_admin_can_delete_any_document(self):
        doc = _make_doc(id=10, owner_id=99)
        self.doc_repo.get_by_id.return_value = doc

        self.use_case.execute(document_id=10, user_id=1, user_role="admin")

        self.doc_repo.delete.assert_called_once_with(10)

    def test_non_owner_non_admin_cannot_delete(self):
        doc = _make_doc(id=10, owner_id=5)
        self.doc_repo.get_by_id.return_value = doc

        with pytest.raises(BusinessRuleViolation) as exc_info:
            self.use_case.execute(document_id=10, user_id=99, user_role="user")
        assert "Can only delete" in str(exc_info.value)

    def test_nonexistent_document_raises(self):
        self.doc_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFound) as exc_info:
            self.use_case.execute(document_id=999, user_id=1, user_role="admin")
        assert "Document" in str(exc_info.value)
        assert "999" in str(exc_info.value)

    def test_delete_without_source_path_skips_file_deletion(self):
        doc = _make_doc(id=10, owner_id=5, source_path="")
        self.doc_repo.get_by_id.return_value = doc

        self.use_case.execute(document_id=10, user_id=5, user_role="user")

        self.file_storage.delete_file.assert_not_called()
        self.doc_repo.delete.assert_called_once_with(10)

    def test_vector_store_deleted_before_document(self):
        """Verify order: vector store first, then file, then doc record."""
        doc = _make_doc(id=10, owner_id=5)
        self.doc_repo.get_by_id.return_value = doc

        call_order = []
        self.vector_store_repo.delete_by_document_id.side_effect = lambda x: call_order.append("vector")
        self.file_storage.delete_file.side_effect = lambda x: call_order.append("file")
        self.doc_repo.delete.side_effect = lambda x: call_order.append("doc")

        self.use_case.execute(document_id=10, user_id=5, user_role="user")

        assert call_order == ["vector", "file", "doc"]


# ---------------------------------------------------------------------------
# GetDocument
# ---------------------------------------------------------------------------


class TestGetDocument:
    def setup_method(self):
        self.doc_repo = MagicMock()
        self.group_repo = MagicMock()
        self.client_assignment_repo = MagicMock()
        self.use_case = GetDocument(self.doc_repo, self.group_repo, self.client_assignment_repo)

    def test_internal_user_views_public_document(self):
        doc = _make_doc(visibility=DocumentVisibility.INTERNAL_PUBLIC)
        self.doc_repo.get_by_id.return_value = doc

        result = self.use_case.execute(document_id=1, user_id=1, user_kind="internal", user_role="user")

        assert result.id == 1
        assert result.filename == "doc.pdf"

    def test_nonexistent_document_raises(self):
        self.doc_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFound):
            self.use_case.execute(document_id=999, user_id=1, user_kind="internal", user_role="user")

    def test_internal_user_views_own_private_document(self):
        doc = _make_doc(id=5, visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=1)
        self.doc_repo.get_by_id.return_value = doc

        result = self.use_case.execute(document_id=5, user_id=1, user_kind="internal", user_role="user")
        assert result.id == 5

    def test_internal_user_rejected_from_other_private_document(self):
        doc = _make_doc(id=5, visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=99)
        self.doc_repo.get_by_id.return_value = doc
        self.group_repo.get_user_group_ids.return_value = []
        self.client_assignment_repo.get_assigned_client_ids.return_value = []

        with pytest.raises(BusinessRuleViolation) as exc_info:
            self.use_case.execute(document_id=5, user_id=1, user_kind="internal", user_role="user")
        assert "No access" in str(exc_info.value)

    def test_client_user_views_own_client_private_document(self):
        doc = _make_doc(id=5, visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=50)
        self.doc_repo.get_by_id.return_value = doc

        result = self.use_case.execute(document_id=5, user_id=50, user_kind="client", user_role="user")
        assert result.id == 5

    def test_client_user_rejected_from_other_client_document(self):
        doc = _make_doc(id=5, visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=51)
        self.doc_repo.get_by_id.return_value = doc

        with pytest.raises(BusinessRuleViolation):
            self.use_case.execute(document_id=5, user_id=50, user_kind="client", user_role="user")

    def test_internal_user_views_assigned_client_document(self):
        doc = _make_doc(id=5, visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=50)
        self.doc_repo.get_by_id.return_value = doc
        self.client_assignment_repo.get_assigned_client_ids.return_value = [50, 51]

        result = self.use_case.execute(document_id=5, user_id=1, user_kind="internal", user_role="user")
        assert result.id == 5

    def test_internal_user_rejected_from_unassigned_client_document(self):
        doc = _make_doc(id=5, visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=50)
        self.doc_repo.get_by_id.return_value = doc
        self.client_assignment_repo.get_assigned_client_ids.return_value = [99]

        with pytest.raises(BusinessRuleViolation):
            self.use_case.execute(document_id=5, user_id=1, user_kind="internal", user_role="user")

    def test_internal_user_views_group_document_as_member(self):
        doc = _make_doc(id=5, visibility=DocumentVisibility.INTERNAL_GROUP, group_id=10)
        self.doc_repo.get_by_id.return_value = doc
        self.group_repo.get_user_group_ids.return_value = [10, 20]

        result = self.use_case.execute(document_id=5, user_id=1, user_kind="internal", user_role="user")
        assert result.id == 5

    def test_internal_user_rejected_from_group_document_as_non_member(self):
        doc = _make_doc(id=5, visibility=DocumentVisibility.INTERNAL_GROUP, group_id=10)
        self.doc_repo.get_by_id.return_value = doc
        self.group_repo.get_user_group_ids.return_value = [1, 2]
        self.client_assignment_repo.get_assigned_client_ids.return_value = []

        with pytest.raises(BusinessRuleViolation):
            self.use_case.execute(document_id=5, user_id=1, user_kind="internal", user_role="user")

    def test_client_user_rejected_from_internal_group_document(self):
        doc = _make_doc(id=5, visibility=DocumentVisibility.INTERNAL_GROUP, group_id=10)
        self.doc_repo.get_by_id.return_value = doc

        with pytest.raises(BusinessRuleViolation):
            self.use_case.execute(document_id=5, user_id=1, user_kind="client", user_role="user")

    def test_client_user_rejected_from_internal_public(self):
        doc = _make_doc(id=5, visibility=DocumentVisibility.INTERNAL_PUBLIC)
        self.doc_repo.get_by_id.return_value = doc

        with pytest.raises(BusinessRuleViolation):
            self.use_case.execute(document_id=5, user_id=1, user_kind="client", user_role="user")


# ---------------------------------------------------------------------------
# ListDocuments
# ---------------------------------------------------------------------------


class TestListDocuments:
    def setup_method(self):
        self.doc_repo = MagicMock()
        self.group_repo = MagicMock()
        self.client_assignment_repo = MagicMock()
        self.use_case = ListDocuments(self.doc_repo, self.group_repo, self.client_assignment_repo)

    def test_internal_user_gets_visible_documents(self):
        docs = [
            _make_doc(id=1, filename="a.pdf"),
            _make_doc(id=2, filename="b.pdf"),
        ]
        self.group_repo.get_user_group_ids.return_value = [10]
        self.client_assignment_repo.get_assigned_client_ids.return_value = [50]
        self.doc_repo.list_visible.return_value = docs

        result = self.use_case.execute(user_id=1, user_kind="internal")

        assert len(result) == 2
        self.doc_repo.list_visible.assert_called_once_with(
            user_kind="internal",
            user_id=1,
            group_ids=[10],
            assigned_client_ids=[50],
        )

    def test_client_user_gets_visible_documents(self):
        docs = [_make_doc(id=1, filename="c.pdf")]
        self.doc_repo.list_visible.return_value = docs

        result = self.use_case.execute(user_id=50, user_kind="client")

        assert len(result) == 1
        # Client should not query groups or assignments
        self.group_repo.get_user_group_ids.assert_not_called()
        self.client_assignment_repo.get_assigned_client_ids.assert_not_called()
        self.doc_repo.list_visible.assert_called_once_with(
            user_kind="client",
            user_id=50,
            group_ids=[],
            assigned_client_ids=[],
        )

    def test_empty_result(self):
        self.group_repo.get_user_group_ids.return_value = []
        self.client_assignment_repo.get_assigned_client_ids.return_value = []
        self.doc_repo.list_visible.return_value = []

        result = self.use_case.execute(user_id=1, user_kind="internal")
        assert result == []
