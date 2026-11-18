"""SQLAlchemy chunk corpus operations using the caller's session."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from infrastructure.database.models import ChunkModel
from infrastructure.repositories.chunk.chunk_mapping import to_chunk_search_result


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession

    from domain.knowledge.chunk import ChunkRecord


class SQLAlchemyChunkCorpusRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def iter_all_contents_with_acl(
        self, batch_size: int = 1000
    ) -> AsyncIterator[tuple[str, str, int | None, int | None]]:
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

    async def list_space_filenames(self, space_id: int) -> dict[int, str]:
        rows = await self._session.execute(
            select(ChunkModel.document_id, ChunkModel.filename)
            .where(ChunkModel.space_id == space_id)
            .distinct(ChunkModel.document_id)
            .order_by(ChunkModel.document_id)
        )
        return dict(rows.tuples().all())

    async def list_document_chunks(self, document_ids: list[int]) -> list[ChunkRecord]:
        if not document_ids:
            return []

        rows = await self._session.execute(
            select(ChunkModel)
            .where(ChunkModel.document_id.in_(document_ids))
            .order_by(ChunkModel.document_id, ChunkModel.chunk_index)
        )
        return [to_chunk_search_result(chunk) for chunk in rows.scalars().all()]
