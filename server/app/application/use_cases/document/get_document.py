"""Use Case: GetDocument — get document by ID with access check."""

from __future__ import annotations

from domain.exceptions import EntityNotFound
from domain.repositories.client_assignment_repository import ClientAssignmentRepository
from domain.repositories.document_repository import DocumentRepository
from domain.repositories.group_repository import GroupRepository
from domain.services.access_control import can_view_document

from application.dto.document_dto import DocumentDTO


class GetDocument:
    def __init__(
        self,
        document_repo: DocumentRepository,
        group_repo: GroupRepository,
        client_assignment_repo: ClientAssignmentRepository,
    ) -> None:
        self._document_repo = document_repo
        self._group_repo = group_repo
        self._client_assignment_repo = client_assignment_repo

    def execute(self, document_id: int, user_id: int, user_kind: str, user_role: str) -> DocumentDTO:
        doc = self._document_repo.get_by_id(document_id)
        if doc is None:
            raise EntityNotFound("Document", document_id)

        user_group_ids = self._group_repo.get_user_group_ids(user_id) if user_kind == "internal" else []
        assigned_ids = (
            self._client_assignment_repo.get_assigned_client_ids(user_id) if user_kind == "internal" else []
        )

        if not can_view_document(
            doc_visibility=doc.visibility,
            doc_owner_id=doc.owner_id,
            doc_group_id=doc.group_id,
            user_kind=user_kind,
            user_id=user_id,
            user_group_ids=user_group_ids,
            assigned_client_ids=assigned_ids,
        ):
            from domain.exceptions import BusinessRuleViolation

            raise BusinessRuleViolation("No access to this document")

        return DocumentDTO(
            id=doc.id,
            filename=doc.filename,
            visibility=doc.visibility,
            status=doc.status,
            error_message=doc.error_message,
            chunks=doc.chunks,
            chars=doc.chars,
        )
