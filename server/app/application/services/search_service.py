"""Application service for exact substring search with ACL filtering."""

from __future__ import annotations

import logging

from domain.repositories.chunk_repository import ChunkSearchResult
from domain.value_objects.roles import UserKind

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
        user: dict,
        limit: int = 20,
        mode: str = "exact",
        document_id: int | None = None,
    ) -> list[ChunkSearchResult]:
        async with self._uow_factory.create() as uow:
            user_kind = user.get("kind", UserKind.INTERNAL)
            user_role = user.get("role", "user")
            ctx = await self._user_ctx_factory.build(uow, user["id"], user_kind, user_role)
            return await uow.chunks.search_substring(
                query=query,
                user=user,
                group_ids=ctx.group_ids,
                limit=limit,
                mode=mode,
                document_id=document_id,
                managed_client_ids=ctx.managed_client_ids,
                managed_internal_ids=ctx.managed_internal_ids,
                managed_group_ids=ctx.managed_group_ids,
            )
