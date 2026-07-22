"""Use Case: ListDocuments — list documents visible to the user."""

from __future__ import annotations

from domain.repositories.client_assignment_repository import ClientAssignmentRepository
from domain.repositories.document_repository import DocumentRepository
from domain.repositories.group_repository import GroupRepository

from application.dto.document_dto import DocumentDTO


class ListDocuments:
    def __init__(
        self,
        document_repo: DocumentRepository,
        group_repo: GroupRepository,
        client_assignment_repo: ClientAssignmentRepository,
    ) -> None:
        self._document_repo = document_repo
        self._group_repo = group_repo
        self._client_assignment_repo = client_assignment_repo

    def execute(self, user_id: int, user_kind: str) -> list[DocumentDTO]:
        if user_kind == "client":
            group_ids = []
            assigned_ids = []
        else:
            group_ids = self._group_repo.get_user_group_ids(user_id)
            assigned_ids = self._client_assignment_repo.get_assigned_client_ids(user_id)

        docs = self._document_repo.list_visible(
            user_kind=user_kind,
            user_id=user_id,
            group_ids=group_ids or [],
            assigned_client_ids=assigned_ids or [],
        )

        return [
            DocumentDTO(
                id=d.id,
                filename=d.filename,
                visibility=d.visibility,
                status=d.status,
                error_message=d.error_message,
                chunks=d.chunks,
                chars=d.chars,
            )
            for d in docs
        ]
