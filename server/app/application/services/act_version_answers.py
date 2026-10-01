"""Invalidate answers referencing documents of an affected regulatory act."""

from __future__ import annotations

from application.ports.cache_invalidator import CacheInvalidatorPort
from application.ports.unit_of_work_factory import UnitOfWorkFactory


class ActVersionAnswerInvalidator:
    def __init__(
        self, uow_factory: UnitOfWorkFactory, cache_invalidator: CacheInvalidatorPort | None
    ) -> None:
        self._uow_factory = uow_factory
        self._cache_invalidator = cache_invalidator

    async def invalidate_act_answers(self, act_id: int) -> None:
        if self._cache_invalidator is None:
            return
        async with self._uow_factory.create(master=True) as uow:
            versions = await uow.act_versions.list_by_act(act_id)
            document_ids = list({v.document_id for v in versions})
        if document_ids:
            await self._cache_invalidator.invalidate_by_document_ids(document_ids)

    async def invalidate_document_answers(self, document_id: int) -> None:
        if self._cache_invalidator is not None:
            await self._cache_invalidator.invalidate_by_document_ids([document_id])
