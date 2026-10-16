"""Stable facade for document upload, deletion and rename commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

from application.documents.delete import DocumentDeleteCommand
from application.documents.rename import DocumentRenameCommand
from application.documents.upload import DocumentUploadCommand
from application.identity import UserContextFactory
from domain.value_objects.roles import UserKind


if TYPE_CHECKING:
    from application.documents.dto import DocumentDTO, UploadOptions
    from application.ports.bm25_index import BM25IndexPort
    from application.ports.file_storage import FileStorage
    from application.uow import UnitOfWorkFactory
    from domain.domain_profile.registry import DomainProfileRegistry
    from domain.repositories.vector_store_repository import VectorStoreRepository


class DocumentCommandService:
    """Delegate each write use case to a command with its own transaction boundary."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        vector_store_repo: VectorStoreRepository,
        file_storage: FileStorage,
        bm25_index: BM25IndexPort,
        *,
        domain_registry: DomainProfileRegistry | None = None,
        user_ctx_factory: UserContextFactory | None = None,
    ) -> None:
        # Keep the vector-store argument for constructor compatibility; writes use the outbox.
        self._vector_store = vector_store_repo
        context_factory = user_ctx_factory or UserContextFactory()
        self._upload = DocumentUploadCommand(uow_factory, file_storage, context_factory, domain_registry)
        self._delete = DocumentDeleteCommand(uow_factory, file_storage, bm25_index, context_factory)
        self._rename = DocumentRenameCommand(uow_factory, file_storage, context_factory)

    async def upload(
        self,
        filename: str,
        file_data: bytes,
        visibility: str,
        group_id: int | None,
        **options: Unpack[UploadOptions],
    ) -> DocumentDTO:
        return await self._upload.execute(
            filename=filename,
            file_data=file_data,
            visibility=visibility,
            group_id=group_id,
            **options,
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
