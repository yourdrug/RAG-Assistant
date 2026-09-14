"""Adapters — bridge infrastructure implementations to application ports."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    async def search_substring(self, query, user, group_ids, limit=20, mode="exact"):
        async with self._uow_factory.create() as uow:
            return await uow.chunks.search_substring(
                query=query,
                user=user,
                group_ids=group_ids,
                limit=limit,
                mode=mode,
            )

    async def get_neighbors(self, document_id, center_index, window=1, exclude_hashes=None,
                            user=None, group_ids=None):
        async with self._uow_factory.create() as uow:
            return await uow.chunks.get_neighbors(
                document_id=document_id,
                center_index=center_index,
                window=window,
                exclude_hashes=exclude_hashes,
                user=user,
                group_ids=group_ids,
            )

    async def get_table_batches(self, document_id, anchor_index, exclude_hashes=None,
                                user=None, group_ids=None):
        async with self._uow_factory.create() as uow:
            return await uow.chunks.get_table_batches(
                document_id=document_id,
                anchor_index=anchor_index,
                exclude_hashes=exclude_hashes,
                user=user,
                group_ids=group_ids,
            )
