"""Adapters — bridge infrastructure implementations to application ports."""

from __future__ import annotations

from domain.value_objects.chunk_context import TABLE_CONTEXT_MAX_CHUNKS

from typing import TYPE_CHECKING

from domain.repositories.chunk_repository import ChunkRetrievalRepository

if TYPE_CHECKING:
    from infrastructure.uow_factory import UnitOfWorkFactory


class ChunkSearchAdapter:
    """Bridges UoW.chunks to the ChunkSearchPort expected by RagService.

    Lives in infrastructure because it depends on UoW (infrastructure concern).
    The adapter pattern allows the application layer to depend on a port
    without knowing about the concrete UoW implementation.
    """

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def search_substring(self, query, user, limit=20, mode="exact", as_of_date=None):
        async with self._uow_factory.create() as uow:
            repository: ChunkRetrievalRepository = uow.chunks
            return await repository.search_substring(
                query=query,
                user=user,
                limit=limit,
                mode=mode,
                as_of_date=as_of_date,
            )

    async def get_neighbors(self, document_id, center_index, window=1, exclude_hashes=None, *, user):
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
        document_id,
        table_id,
        exclude_hashes=None,
        *,
        user,
        limit=TABLE_CONTEXT_MAX_CHUNKS,
        row_start=None,
        row_end=None,
    ):
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
