"""Replace pack-owned chunk metadata using the caller's session."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update

from infrastructure.database.models import ChunkModel, StagedChunkModel


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class SQLAlchemyChunkPackMetadataRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def set_pack_metadata(self, document_id: int, pack_metadata: dict[str, Any]) -> int:
        result = await self._session.execute(
            update(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .values(pack_metadata=pack_metadata)
        )
        # A staged revision is published later and must not restore older pack state.
        await self._session.execute(
            update(StagedChunkModel)
            .where(StagedChunkModel.document_id == document_id)
            .values(pack_metadata=pack_metadata)
        )
        await self._session.flush()
        return result.rowcount or 0

    async def get_pack_metadata(self, document_id: int) -> dict[str, Any] | None:
        result = await self._session.execute(
            select(ChunkModel.pack_metadata)
            .where(ChunkModel.document_id == document_id)
            .order_by(ChunkModel.chunk_index)
            .limit(1)
        )
        row = result.first()
        return dict(row[0] or {}) if row is not None else None
