"""Application service for exact substring search with ACL filtering."""

from __future__ import annotations

import logging

from domain.repositories.chunk_repository import ChunkSearchResult
from domain.value_objects.roles import UserKind

from application.ports.unit_of_work_factory import UnitOfWorkFactory

log = logging.getLogger("default")


class SearchService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def exact_search(
        self,
        query: str,
        user: dict,
        limit: int = 20,
        mode: str = "exact",
        document_id: int | None = None,
    ) -> list[ChunkSearchResult]:
        async with self._uow_factory.create() as uow:
            group_ids = await uow.groups.get_user_group_ids(user["id"])
            assigned_client_ids = (
                await uow.client_assignments.get_assigned_client_ids(user["id"])
                if user["kind"] == UserKind.INTERNAL.value
                else []
            )
            return await uow.chunks.search_substring(
                query=query,
                user=user,
                group_ids=group_ids,
                assigned_client_ids=assigned_client_ids,
                limit=limit,
                mode=mode,
                document_id=document_id,
            )
