"""ChunkQueryService -- list/query chunk operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.chunks.dto import ChunkCursorQuery
from application.chunks.mappers import to_chunk_dto
from application.identity import UserContextFactory
from domain.services import check_document_access
from domain.shared.exceptions import BusinessRuleViolation, EntityNotFound
from domain.shared.pagination import CursorPage
from domain.utils import decode_cursor


if TYPE_CHECKING:
    from application.chunks.dto import ChunkItemDTO
    from application.uow import UnitOfWorkFactory
    from domain.entities.document import Document
    from domain.identity.roles import UserKind, UserRole
    from domain.repositories.chunk_repository import ChunkSearchResult

log = logging.getLogger(__name__)


def validate_chunk_acl(document: Document, chunks: list[ChunkSearchResult]) -> None:
    # Fail closed on legacy inconsistency or an ACL change between SQL reads.

    for chunk in chunks:
        if (
            chunk.document_id != document.id
            or chunk.visibility != document.visibility
            or chunk.owner_id != document.owner_id
            or chunk.group_id != document.group_id
        ):
            msg = "Chunk ACL does not match document ACL"
            raise BusinessRuleViolation(msg)


class ChunkQueryService:
    """List/query chunk operations."""

    def __init__(
        self, uow_factory: UnitOfWorkFactory, user_ctx_factory: UserContextFactory | None = None
    ) -> None:
        self._uow_factory = uow_factory
        self._user_ctx_factory = user_ctx_factory or UserContextFactory()

    async def list_chunks(
        self,
        document_id: int,
        user_id: int,
        user_kind: UserKind,
        user_role: UserRole,
        *,
        limit: int = 50,
        offset: int = 0,
        content_hashes: list[str] | None = None,
    ) -> tuple[list[ChunkItemDTO], int]:
        """List chunks for a document with pagination."""
        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)

            if doc is None:
                msg = "Document"
                raise EntityNotFound(msg, document_id)

            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            check_document_access(doc, ctx)

            chunks, total = await uow.chunks.list_for_document(
                document_id,
                limit=limit,
                offset=offset,
                content_hashes=content_hashes,
            )
            validate_chunk_acl(doc, chunks)

            return [to_chunk_dto(c) for c in chunks], total

    async def list_chunks_cursor(
        self,
        document_id: int,
        user_id: int,
        user_kind: UserKind,
        user_role: UserRole,
        *,
        query: ChunkCursorQuery | None = None,
    ) -> CursorPage[ChunkItemDTO]:
        """List chunks for a document with cursor-based pagination."""
        query = query or ChunkCursorQuery()
        decoded = decode_cursor(query.cursor) if query.cursor else None

        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)

            if doc is None:
                msg = "Document"
                raise EntityNotFound(msg, document_id)

            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            check_document_access(doc, ctx)

            page = await uow.chunks.list_for_document_cursor(
                document_id,
                limit=query.limit,
                cursor=decoded,
                direction=query.direction,
                content_hashes=query.content_hashes,
            )
            validate_chunk_acl(doc, page.items)

            return CursorPage(
                items=[to_chunk_dto(c) for c in page.items],
                next_cursor=page.next_cursor,
                prev_cursor=page.prev_cursor,
            )
