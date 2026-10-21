"""Document chunk cursor round trips against PostgreSQL."""

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from domain.knowledge.chunk import decode_cursor
from domain.knowledge.enums import DocumentVisibility
from infrastructure.database.models import ChunkModel, DocumentModel
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.parametrize("anchor", [3, 6])
async def test_backward_page_continues_forward_without_repeats(pg_engine: AsyncEngine, anchor: int) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        doc = DocumentModel(filename="cursor.txt", visibility=DocumentVisibility.INTERNAL_PUBLIC.value)
        session.add(doc)
        await session.flush()
        session.add_all(
            ChunkModel(
                document_id=doc.id,
                chunk_index=i,
                content=str(i),
                visibility=DocumentVisibility.INTERNAL_PUBLIC.value,
            )
            for i in range(8)
        )
    async with sessions() as session:
        repository = SQLAlchemyChunkRepository(session)
        # Anchor IDs are loaded through the same forward listing API.
        all_chunks = await repository.list_for_document_cursor(doc.id, limit=10)
        anchor_chunk = all_chunks.items[anchor]
        previous = await repository.list_for_document_cursor(
            doc.id, limit=3, cursor=(anchor_chunk.chunk_index, anchor_chunk.chunk_id), direction="prev"
        )
        assert [item.chunk_index for item in previous.items] == list(range(anchor - 3, anchor))
        assert previous.next_cursor is not None
        assert decode_cursor(previous.next_cursor) == (
            previous.items[-1].chunk_index,
            previous.items[-1].chunk_id,
        )
        following = await repository.list_for_document_cursor(
            doc.id, limit=3, cursor=decode_cursor(previous.next_cursor)
        )
        assert [item.chunk_index for item in following.items] == list(range(anchor, min(anchor + 3, 8)))

        if anchor == 3:
            assert previous.prev_cursor is None
        else:
            assert previous.prev_cursor is not None
            assert decode_cursor(previous.prev_cursor)[0] == anchor - 3
