"""Application service for exact substring search with ACL filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.identity import UserContextFactory
from application.spaces.resolution import client_user, document_space, resolve_space
from domain.identity.roles import UserKind
from domain.knowledge.enums import SearchMode


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from datetime import date

    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.identity.user_context import UserContext
    from domain.knowledge.chunk import ChunkRecord
    from domain.knowledge.pack_metadata import MetadataCondition
    from domain.knowledge.space import KnowledgeSpace

    type ChunkSelection = Callable[[date | None], MetadataCondition | None]


class SearchService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        user_ctx_factory: UserContextFactory | None = None,
        chunk_selections: Mapping[str, ChunkSelection] | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._user_ctx_factory = user_ctx_factory or UserContextFactory()
        # Each pack's selection (e.g. editions in force today) by pack key; a missing key keeps all chunks.
        self._chunk_selections = dict(chunk_selections or {})

    async def exact_search(
        self,
        query: str,
        user: UserContext,
        *,
        limit: int = 20,
        mode: str = SearchMode.EXACT.value,
        document_id: int | None = None,
        offset: int = 0,
        space_id: int | None = None,
    ) -> tuple[list[ChunkRecord], int]:
        """Search one space: the given one, the space of ``document_id``, or the only space.

        A client searches only its support spaces: the given one or its only one.
        """
        async with self._uow_factory.create() as uow:
            full_ctx = await self._user_ctx_factory.build(uow, user.user_id, user.user_kind, user.user_role)
            client = await client_user(uow, user.user_id) if full_ctx.user_kind == UserKind.CLIENT else None

            # A client searches only its support spaces, like it asks only in them.
            if client is not None and document_id is None:
                space_id = client.choose_client_space(space_id)

            space = await self.search_space(uow, space_id, document_id)

            if space is None or (client is not None and space.id not in client.client_space_ids):
                return [], 0

            selection = self._chunk_selections.get(space.pack_key)
            return await uow.chunks.search_substring_page(
                query=query,
                user=full_ctx,
                limit=limit,
                offset=offset,
                mode=mode,
                document_id=document_id,
                selection=selection(None) if selection is not None else None,
                space_id=space.id,
            )

    @staticmethod
    async def search_space(
        uow: UnitOfWork, space_id: int | None, document_id: int | None
    ) -> KnowledgeSpace | None:
        if space_id is None and document_id is not None:
            document = await uow.documents.get_by_id(document_id)
            # A missing document matches nothing; the search itself still applies ACL.
            return await document_space(uow, document) if document is not None else None

        return await resolve_space(uow, space_id)
