"""Adapters — bridge infrastructure implementations to application ports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.knowledge.chunk import TABLE_CONTEXT_MAX_CHUNKS


if TYPE_CHECKING:
    from datetime import date

    from application.uow.factory import UnitOfWorkFactory
    from domain.identity.user_context import UserContext
    from domain.knowledge.chunk import ChunkRecord
    from domain.rag.ports import ChunkRetrievalRepository


class ChunkSearchAdapter:
    """Bridges UoW.chunks to the ChunkSearchPort expected by RagService.

    Lives in infrastructure because it depends on UoW (infrastructure concern).
    The adapter pattern allows the application layer to depend on a port
    without knowing about the concrete UoW implementation.
    """

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def search_substring(
        self,
        query: str,
        user: UserContext,
        *,
        limit: int = 20,
        mode: str = "exact",
        as_of_date: date | None = None,
        document_id: int | None = None,
    ) -> list[ChunkRecord]:
        async with self._uow_factory.create() as uow:
            repository: ChunkRetrievalRepository = uow.chunks
            return await repository.search_substring(
                query=query,
                user=user,
                limit=limit,
                mode=mode,
                as_of_date=as_of_date,
                document_id=document_id,
            )

    async def get_neighbors(
        self,
        document_id: int,
        center_index: int,
        window: int = 1,
        exclude_hashes: set[str] | None = None,
        *,
        user: UserContext,
    ) -> list[ChunkRecord]:
        async with self._uow_factory.create() as uow:
            repository: ChunkRetrievalRepository = uow.chunks
            return await repository.get_neighbors(
                document_id=document_id,
                center_index=center_index,
                window=window,
                exclude_hashes=exclude_hashes,
                user=user,
            )

    async def get_table_batches(
        self,
        document_id: int,
        table_id: str,
        exclude_hashes: set[str] | None = None,
        *,
        user: UserContext,
        limit: int = TABLE_CONTEXT_MAX_CHUNKS,
        row_start: int | None = None,
        row_end: int | None = None,
    ) -> list[ChunkRecord]:
        async with self._uow_factory.create() as uow:
            return await uow.chunks.get_table_batches(
                document_id,
                table_id,
                exclude_hashes,
                user=user,
                limit=limit,
                row_start=row_start,
                row_end=row_end,
            )
