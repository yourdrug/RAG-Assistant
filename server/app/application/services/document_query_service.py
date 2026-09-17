"""DocumentQueryService — list, get, search (read operations)."""

from __future__ import annotations

import logging
from dataclasses import replace

from application.dto.document_dto import ClientInfo, DocumentDTO
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.user_context_factory import UserContextFactory
from domain.exceptions import EntityNotFound
from domain.services import check_document_access, is_in_search_scope
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserKind, UserRole

log = logging.getLogger(__name__)


class DocumentQueryService:
    """Read operations: list, get, search."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        user_ctx_factory: UserContextFactory | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._user_ctx_factory = user_ctx_factory or UserContextFactory()

    async def _enrich_with_outbox_status(self, uow, dto: DocumentDTO) -> DocumentDTO:
        if dto.status != DocumentStatus.INDEXING.value:
            return dto
        outbox = await uow.vector_outbox.count_by_document(dto.id)
        failed = None
        if outbox["failed"] > 0:
            failed = await uow.vector_outbox.get_failed_details(dto.id)
        return replace(
            dto,
            outbox_pending=outbox["pending"],
            outbox_failed=outbox["failed"],
            outbox_failed_details=failed,
        )

    async def list_documents(
        self,
        user_id: int,
        user_kind: str,
        user_role: str | UserRole = UserRole.USER,
        limit: int = 200,
        offset: int = 0,
    ) -> list[DocumentDTO]:
        async with self._uow_factory.create() as uow:
            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            if ctx.is_admin:
                docs = await uow.documents.list_all(limit=limit, offset=offset)
                dtos = [DocumentDTO.from_entity(d, in_search_scope=is_in_search_scope(d, ctx)) for d in docs]
            elif ctx.is_curator:
                docs = await uow.documents.list_visible(
                    user_kind=user_kind,
                    user_id=user_id,
                    group_ids=list(ctx.group_ids) or [],
                    user_role=user_role,
                    limit=limit,
                    offset=offset,
                    managed_client_ids=list(ctx.managed_client_ids),
                    managed_internal_ids=list(ctx.managed_internal_ids),
                    managed_group_ids=list(ctx.managed_group_ids),
                )
                dtos = [DocumentDTO.from_entity(d) for d in docs]
            elif ctx.is_client:
                docs = await uow.documents.list_visible(
                    user_kind=user_kind,
                    user_id=user_id,
                    group_ids=[],
                    user_role=user_role,
                    limit=limit,
                    offset=offset,
                )
                dtos = [DocumentDTO.from_entity(d) for d in docs]
            else:
                docs = await uow.documents.list_visible(
                    user_kind=user_kind,
                    user_id=user_id,
                    group_ids=list(ctx.group_ids) or [],
                    user_role=user_role,
                    limit=limit,
                    offset=offset,
                )
                dtos = [DocumentDTO.from_entity(d) for d in docs]
            return [await self._enrich_with_outbox_status(uow, dto) for dto in dtos]

    async def get_document(
        self, document_id: int, user_id: int, user_kind: str, user_role: str
    ) -> DocumentDTO:
        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)
            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            check_document_access(doc, ctx)
            dto = DocumentDTO.from_entity(doc)
            return await self._enrich_with_outbox_status(uow, dto)

    async def list_uploadable_clients(self, user_id: int, user_kind: str, user_role: str) -> list[ClientInfo]:
        async with self._uow_factory.create() as uow:
            if user_kind == UserKind.CLIENT:
                return []
            if user_role == UserRole.ADMIN:
                all_users = await uow.users.list_all()
                return [
                    ClientInfo(id=u.id, email=u.email)
                    for u in all_users
                    if u.kind == UserKind.CLIENT and u.id is not None
                ]
            if user_role == UserRole.CURATOR:
                managed_client_ids = await uow.assignments.get_managed_client_ids(user_id)
                if not managed_client_ids:
                    return []
                all_users = await uow.users.list_all()
                return [
                    ClientInfo(id=u.id, email=u.email)
                    for u in all_users
                    if u.kind == UserKind.CLIENT and u.id is not None and u.id in managed_client_ids
                ]
            return []

    async def list_source_files(self, search: str | None = None) -> list[str]:
        async with self._uow_factory.create() as uow:
            return await uow.documents.list_distinct_filenames(search=search, limit=100)
