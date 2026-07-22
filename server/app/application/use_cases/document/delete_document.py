"""Use Case: DeleteDocument — delete document with ownership check."""

from __future__ import annotations

from domain.exceptions import BusinessRuleViolation, EntityNotFound
from domain.repositories.document_repository import DocumentRepository
from domain.repositories.group_repository import GroupRepository
from domain.value_objects.roles import UserRole


class DeleteDocument:
    def __init__(
        self,
        document_repo: DocumentRepository,
        vector_store_repo,
        file_storage,
        group_repo: GroupRepository | None = None,
    ) -> None:
        self._document_repo = document_repo
        self._vector_store_repo = vector_store_repo
        self._file_storage = file_storage
        self._group_repo = group_repo

    def execute(self, document_id: int, user_id: int, user_role: str) -> None:
        doc = self._document_repo.get_by_id(document_id)
        if doc is None:
            raise EntityNotFound("Document", document_id)

        role = UserRole(user_role)
        user_group_ids = None
        if self._group_repo is not None:
            user_group_ids = self._group_repo.get_user_group_ids(user_id)
        if not doc.can_be_deleted_by(user_id, role, user_group_ids):
            raise BusinessRuleViolation("Can only delete your own documents")

        self._vector_store_repo.delete_by_document_id(document_id)

        if doc.source_path:
            self._file_storage.delete_file(doc.source_path)

        self._document_repo.delete(document_id)
