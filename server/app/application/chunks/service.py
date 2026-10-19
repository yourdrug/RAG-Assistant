"""ChunkService -- thin facade for chunk lifecycle management.

Delegates to ChunkQueryService, ChunkMutationService, and ManualDocumentService.
The delegated services manage their own async UnitOfWork boundaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack


if TYPE_CHECKING:
    from application.chunks.dto import AddChunkResult, ChunkItemDTO, EditChunkResult, ListChunksCursorOptions
    from application.chunks.mutations import ChunkMutationService
    from application.chunks.queries import ChunkQueryService
    from application.documents import DocumentDTO, ManualDocumentService
    from domain.value_objects.cursor_page import CursorPage


class ChunkService:
    def __init__(
        self,
        query_service: ChunkQueryService,
        mutation_service: ChunkMutationService,
        manual_document_service: ManualDocumentService,
    ) -> None:
        self._query = query_service
        self._mutation = mutation_service
        self._manual_doc = manual_document_service

    async def list_chunks(
        self,
        document_id: int,
        user_id: int,
        user_kind: str,
        user_role: str,
        *,
        limit: int = 50,
        offset: int = 0,
        content_hashes: list[str] | None = None,
    ) -> tuple[list[ChunkItemDTO], int]:
        return await self._query.list_chunks(
            document_id=document_id,
            user_id=user_id,
            user_kind=user_kind,
            user_role=user_role,
            limit=limit,
            offset=offset,
            content_hashes=content_hashes,
        )

    async def list_chunks_cursor(
        self,
        document_id: int,
        user_id: int,
        user_kind: str,
        user_role: str,
        **options: Unpack[ListChunksCursorOptions],
    ) -> CursorPage[ChunkItemDTO]:
        return await self._query.list_chunks_cursor(
            document_id=document_id,
            user_id=user_id,
            user_kind=user_kind,
            user_role=user_role,
            limit=options.get("limit", 50),
            cursor=options.get("cursor"),
            direction=options.get("direction", "next"),
            content_hashes=options.get("content_hashes"),
        )

    async def edit_chunk(
        self,
        document_id: int,
        chunk_id: int,
        content: str,
        user_id: int,
        user_role: str,
    ) -> EditChunkResult:
        return await self._mutation.edit_chunk(
            document_id=document_id,
            chunk_id=chunk_id,
            content=content,
            user_id=user_id,
            user_role=user_role,
        )

    async def add_chunk(
        self,
        document_id: int,
        content: str,
        user_id: int,
        user_role: str,
        *,
        page: int | None = None,
        section: str | None = None,
    ) -> AddChunkResult:
        return await self._mutation.add_chunk(
            document_id=document_id,
            content=content,
            user_id=user_id,
            user_role=user_role,
            page=page,
            section=section,
        )

    async def delete_chunk(
        self,
        document_id: int,
        chunk_id: int,
        user_id: int,
        user_role: str,
    ) -> None:
        await self._mutation.delete_chunk(
            document_id=document_id,
            chunk_id=chunk_id,
            user_id=user_id,
            user_role=user_role,
        )

    async def create_manual_document(
        self,
        title: str,
        visibility: str,
        user_id: int,
        user_kind: str,
        *,
        user_role: str,
        group_id: int | None = None,
    ) -> DocumentDTO:
        return await self._manual_doc.create_manual_document(
            title=title,
            visibility=visibility,
            user_id=user_id,
            user_kind=user_kind,
            user_role=user_role,
            group_id=group_id,
        )
