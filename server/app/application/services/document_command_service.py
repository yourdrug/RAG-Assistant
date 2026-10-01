"""Stable facade for document upload, deletion and rename commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.dto.document_dto import DocumentDTO
from application.ports.bm25_index import BM25IndexPort
from application.ports.file_storage import FileStorage
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.document_delete_command import DocumentDeleteCommand
from application.services.document_rename_command import DocumentRenameCommand
from application.services.document_upload_command import DocumentUploadCommand
from application.services.user_context_factory import UserContextFactory
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.value_objects.roles import UserKind

if TYPE_CHECKING:
    from domain.domain_profile.registry import DomainProfileRegistry


class DocumentCommandService:
    """Delegate each write use case to a command with its own transaction boundary."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        vector_store_repo: VectorStoreRepository,
        file_storage: FileStorage,
        bm25_index: BM25IndexPort,
        domain_registry: DomainProfileRegistry | None = None,
        user_ctx_factory: UserContextFactory | None = None,
    ) -> None:
        # Keep the vector-store argument for constructor compatibility; writes use the outbox.
        context_factory = user_ctx_factory or UserContextFactory()
        self._upload = DocumentUploadCommand(uow_factory, file_storage, context_factory, domain_registry)
        self._delete = DocumentDeleteCommand(uow_factory, file_storage, bm25_index, context_factory)
        self._rename = DocumentRenameCommand(uow_factory, file_storage, context_factory)

    async def upload(
        self,
        filename,
        file_data,
        visibility,
        group_id,
        user_id,
        user_kind,
        user_role,
        client_id=None,
        rename_on_conflict=False,
        doc_domain=None,
        replaces_document_id=None,
    ) -> DocumentDTO:
        return await self._upload.execute(
            filename=filename,
            file_data=file_data,
            visibility=visibility,
            group_id=group_id,
            user_id=user_id,
            user_kind=user_kind,
            user_role=user_role,
            client_id=client_id,
            rename_on_conflict=rename_on_conflict,
            doc_domain=doc_domain,
            replaces_document_id=replaces_document_id,
        )

    async def delete_document(
        self, document_id: int, user_id: int, user_role: str, user_kind: str = UserKind.INTERNAL
    ) -> None:
        await self._delete.execute(document_id, user_id, user_role, user_kind)

    async def rename_document(
        self,
        document_id: int,
        new_filename: str,
        user_id: int,
        user_role: str,
        user_kind: str = UserKind.INTERNAL,
    ) -> DocumentDTO:
        return await self._rename.execute(document_id, new_filename, user_id, user_role, user_kind)
