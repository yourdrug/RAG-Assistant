"""SQLAlchemy chunk corpus operations using the caller's session."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ChunkModel


class SQLAlchemyChunkCorpusRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def iter_all_contents_with_acl(self, batch_size: int = 1000):
        """Stream chunk content and ACL metadata without retaining earlier rows."""
        stmt = (
            select(
                ChunkModel.content,
                ChunkModel.visibility,
                ChunkModel.owner_id,
                ChunkModel.group_id,
            )
            .order_by(ChunkModel.document_id, ChunkModel.chunk_index)
            .execution_options(yield_per=batch_size)
        )
        result = await self._session.stream(stmt)
        try:
            async for row in result:
                yield row[0], row[1], row[2], row[3]
        finally:
            await result.close()
