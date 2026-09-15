"""ChunkQueryService -- list/query chunk operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.dto.chunk_dto import ChunkItemDTO
from application.services.chunk_mappers import to_chunk_dto
from application.services.user_context_factory import UserContextFactory
from domain.exceptions import EntityNotFound
from domain.services import check_document_access
from domain.utils import decode_cursor
from domain.value_objects.cursor_page import CursorPage

if TYPE_CHECKING:
    from application.ports.unit_of_work_factory import UnitOfWorkFactory

log = logging.getLogger(__name__)


class ChunkQueryService:
    """List/query chunk operations."""

    def __init__(
        self, uow_factory: "UnitOfWorkFactory", user_ctx_factory: UserContextFactory | None = None
    ) -> None:
        self._uow_factory = uow_factory
        self._user_ctx_factory = user_ctx_factory or UserContextFactory()

    async def list_chunks(
        self,
        document_id: int,
        user_id: int,
        user_kind: str,
        user_role: str,
        limit: int = 50,
        offset: int = 0,
        content_hashes: list[str] | None = None,
    ) -> tuple[list[ChunkItemDTO], int]:
        """List chunks for a document with pagination."""
        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            check_document_access(doc, ctx)

            chunks, total = await uow.chunks.list_for_document(
                document_id,
                limit=limit,
                offset=offset,
                content_hashes=content_hashes,
            )

            return [to_chunk_dto(c) for c in chunks], total

    async def list_chunks_cursor(
        self,
        document_id: int,
        user_id: int,
        user_kind: str,
        user_role: str,
        limit: int = 50,
        cursor: str | None = None,
        direction: str = "next",
        content_hashes: list[str] | None = None,
    ) -> CursorPage[ChunkItemDTO]:
        """List chunks for a document with cursor-based pagination."""
        decoded = decode_cursor(cursor) if cursor else None

        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            check_document_access(doc, ctx)

            page = await uow.chunks.list_for_document_cursor(
                document_id,
                limit=limit,
                cursor=decoded,
                direction=direction,
                content_hashes=content_hashes,
            )

            return CursorPage(
                items=[to_chunk_dto(c) for c in page.items],
                next_cursor=page.next_cursor,
                prev_cursor=page.prev_cursor,
            )
