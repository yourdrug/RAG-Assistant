"""SQLAlchemy chunk versioning operations using the caller's session."""

from __future__ import annotations

from datetime import date

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ChunkModel


class SQLAlchemyChunkVersioningRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def set_current_by_act_version_ids(self, act_version_ids: list[int], is_current: bool) -> int:
        if not act_version_ids:
            return 0
        result = await self._session.execute(
            update(ChunkModel)
            .where(ChunkModel.act_version_id.in_(act_version_ids))
            .values(is_current=is_current)
        )
        await self._session.flush()
        return result.rowcount or 0

    async def update_temporal_by_act_version_id(
        self,
        act_version_id: int,
        effective_from: date | None,
        effective_to: date | None,
    ) -> int:
        result = await self._session.execute(
            update(ChunkModel)
            .where(ChunkModel.act_version_id == act_version_id)
            .values(effective_from=effective_from, effective_to=effective_to)
        )
        await self._session.flush()
        return result.rowcount or 0
