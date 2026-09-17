"""Application service for exact substring search with ACL filtering."""

from __future__ import annotations

import logging

from domain.repositories.chunk_repository import ChunkSearchResult
from domain.value_objects.user_context import UserContext

from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.user_context_factory import UserContextFactory

log = logging.getLogger("default")


class SearchService:
    def __init__(
        self, uow_factory: UnitOfWorkFactory, user_ctx_factory: UserContextFactory | None = None
    ) -> None:
        self._uow_factory = uow_factory
        self._user_ctx_factory = user_ctx_factory or UserContextFactory()

    async def exact_search(
        self,
        query: str,
        user: UserContext,
        limit: int = 20,
        mode: str = "exact",
        document_id: int | None = None,
    ) -> list[ChunkSearchResult]:
        async with self._uow_factory.create() as uow:
            full_ctx = await self._user_ctx_factory.build(uow, user.user_id, user.user_kind, user.user_role)
            return await uow.chunks.search_substring(
                query=query,
                user=full_ctx,
                limit=limit,
                mode=mode,
                document_id=document_id,
            )
