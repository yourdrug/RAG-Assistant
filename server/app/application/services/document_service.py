"""DocumentService — facade delegating to DocumentCommandService + DocumentQueryService.

Preserves the original public API for backward compatibility while the
actual logic lives in the focused sub-services.
"""

from __future__ import annotations

from application.dto.document_dto import ClientInfo, DocumentDTO
from application.ports.bm25_index import BM25IndexPort
from application.ports.file_storage import FileStorage
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.document_command_service import DocumentCommandService
from application.services.document_query_service import DocumentQueryService
from application.services.user_context_factory import UserContextFactory


class DocumentService:
    """Thin facade — delegates to DocumentCommandService and DocumentQueryService."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        vector_store_repo,
        file_storage: FileStorage,
        bm25_index: BM25IndexPort,
        domain_registry=None,
        act_versioning_service=None,
        user_ctx_factory: UserContextFactory | None = None,
    ) -> None:
        self._cmd = DocumentCommandService(
            uow_factory=uow_factory,
            vector_store_repo=vector_store_repo,
            file_storage=file_storage,
            bm25_index=bm25_index,
            domain_registry=domain_registry,
            act_versioning_service=act_versioning_service,
            user_ctx_factory=user_ctx_factory,
        )
        self._query = DocumentQueryService(
            uow_factory=uow_factory,
            user_ctx_factory=user_ctx_factory,
        )

    def _get_domain_profile(self, doc_domain: str):
        return self._cmd._get_domain_profile(doc_domain)

    async def _resolve_version_group(self, uow, existing, replaces_document_id, pending_replace_id):
        return await self._cmd._resolve_version_group(uow, existing, replaces_document_id, pending_replace_id)

    async def _persist_upload(self, uow, doc, file_data, owner_id, effective_group_id, filename):
        return await self._cmd._persist_upload(uow, doc, file_data, owner_id, effective_group_id, filename)

    async def _maybe_resolve_conflict_sync(
        self, doc, existing, pending_replace_id, doc_domain, uow, storage_deletes
    ):
        return await self._cmd._maybe_resolve_conflict_sync(
            doc, existing, pending_replace_id, doc_domain, uow, storage_deletes
        )

    @staticmethod
    def _validate_no_active_processing(existing) -> None:
        DocumentCommandService._validate_no_active_processing(existing)

    async def _remove_document_from_bm25(self, uow, document_id: int) -> None:
        return await self._cmd._remove_document_from_bm25(uow, document_id)

    async def _enrich_with_outbox_status(self, uow, dto):
        return await self._query._enrich_with_outbox_status(uow, dto)

    async def upload(
        self,
        filename: str,
        file_data: bytes,
        visibility: str,
        group_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        client_id: int | None = None,
        rename_on_conflict: bool = False,
        doc_domain: str | None = None,
        replaces_document_id: int | None = None,
    ) -> DocumentDTO:
        return await self._cmd.upload(
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
        self, document_id: int, user_id: int, user_role: str, user_kind: str | None = None
    ) -> None:
        if user_kind is not None:
            return await self._cmd.delete_document(document_id, user_id, user_role, user_kind)
        return await self._cmd.delete_document(document_id, user_id, user_role)

    async def rename_document(
        self,
        document_id: int,
        new_filename: str,
        user_id: int,
        user_role: str,
        user_kind: str | None = None,
    ) -> DocumentDTO:
        if user_kind is not None:
            return await self._cmd.rename_document(document_id, new_filename, user_id, user_role, user_kind)
        return await self._cmd.rename_document(document_id, new_filename, user_id, user_role)

    async def list_documents(self, **kwargs) -> list[DocumentDTO]:
        return await self._query.list_documents(**kwargs)

    async def get_document(
        self, document_id: int, user_id: int, user_kind: str, user_role: str
    ) -> DocumentDTO:
        return await self._query.get_document(document_id, user_id, user_kind, user_role)

    async def list_uploadable_clients(self, user_id: int, user_kind: str, user_role: str) -> list[ClientInfo]:
        return await self._query.list_uploadable_clients(user_id, user_kind, user_role)

    async def list_source_files(self, search: str | None = None) -> list[str]:
        return await self._query.list_source_files(search=search)
