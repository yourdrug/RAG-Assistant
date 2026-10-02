"""ChunkService -- thin facade for chunk lifecycle management.

Delegates to ChunkQueryService, ChunkMutationService, and ManualDocumentService.
Each public method opens its own async UnitOfWork via the injected UnitOfWorkFactory.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.dto.chunk_dto import AddChunkResult, ChunkItemDTO, EditChunkResult
from application.dto.document_dto import DocumentDTO
from application.services.chunk_mutation_service import ChunkMutationService
from application.services.chunk_query_service import ChunkQueryService
from application.services.manual_document_service import ManualDocumentService
from domain.value_objects.cursor_page import CursorPage

if TYPE_CHECKING:
    from application.ports.bm25_index import BM25IndexPort
    from application.ports.chunk_settings import ChunkSettingsPort
    from application.ports.unit_of_work_factory import UnitOfWorkFactory
    from application.ports.cache_invalidator import CacheInvalidatorPort
    from domain.repositories.vector_store_repository import VectorStoreRepository

log = logging.getLogger(__name__)


class ChunkService:
    def __init__(
        self,
        uow_factory: "UnitOfWorkFactory",
        vector_store_repo: "VectorStoreRepository",
        chunk_settings: "ChunkSettingsPort",
        bm25_index: "BM25IndexPort",
        chunk_min_len_ratio: float = 0.3,
        chunk_max_len_ratio: float = 2.0,
        cache_invalidator: CacheInvalidatorPort | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._vector_store = vector_store_repo

        self._query = ChunkQueryService(uow_factory)
        self._mutation = ChunkMutationService(
            uow_factory,
            chunk_settings,
            bm25_index,
            chunk_min_len_ratio=chunk_min_len_ratio,
            chunk_max_len_ratio=chunk_max_len_ratio,
            cache_invalidator=cache_invalidator,
        )
        self._manual_doc = ManualDocumentService(uow_factory)

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
        limit: int = 50,
        cursor: str | None = None,
        direction: str = "next",
        content_hashes: list[str] | None = None,
    ) -> CursorPage[ChunkItemDTO]:
        return await self._query.list_chunks_cursor(
            document_id=document_id,
            user_id=user_id,
            user_kind=user_kind,
            user_role=user_role,
            limit=limit,
            cursor=cursor,
            direction=direction,
            content_hashes=content_hashes,
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
