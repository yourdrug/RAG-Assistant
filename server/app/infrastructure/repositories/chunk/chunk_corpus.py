"""SQLAlchemy chunk corpus operations using the caller's session."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import ChunkModel


class SQLAlchemyChunkCorpusRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_all_contents(self) -> list[str]:
        stmt = select(ChunkModel.content).order_by(ChunkModel.document_id, ChunkModel.chunk_index)
        result = await self._session.execute(stmt)
        return [row[0] for row in result.all()]

    async def get_all_contents_batches(self, batch_size: int = 5000) -> list[list[str]]:
        """Load chunk contents in batches to reduce peak memory usage."""
        batches: list[list[str]] = []
        offset = 0
        while True:
            stmt = (
                select(ChunkModel.content)
                .order_by(ChunkModel.document_id, ChunkModel.chunk_index)
                .offset(offset)
                .limit(batch_size)
            )
            result = await self._session.execute(stmt)
            rows = list(result.all())
            if not rows:
                break
            batches.append([row[0] for row in rows])
            if len(rows) < batch_size:
                break
            offset += batch_size
        return batches

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

    async def get_all_contents_batches_with_acl(
        self, batch_size: int = 5000
    ) -> list[list[tuple[str, str, int | None, int | None]]]:
        """Load chunk contents + ACL metadata in batches.

        Returns list of batches, each batch is list of
        (content, visibility, owner_id, group_id) tuples.
        """
        batches: list[list[tuple[str, str, int | None, int | None]]] = []
        offset = 0
        while True:
            stmt = (
                select(
                    ChunkModel.content,
                    ChunkModel.visibility,
                    ChunkModel.owner_id,
                    ChunkModel.group_id,
                )
                .order_by(ChunkModel.document_id, ChunkModel.chunk_index)
                .offset(offset)
                .limit(batch_size)
            )
            result = await self._session.execute(stmt)
            rows = list(result.all())
            if not rows:
                break
            batches.append([(row[0], row[1], row[2], row[3]) for row in rows])
            if len(rows) < batch_size:
                break
            offset += batch_size
        return batches
