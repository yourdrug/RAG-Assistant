"""Cursor listing keeps the same complete records as ordinary listing."""

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from application.chunks.mappers import to_chunk_dto
from domain.knowledge.chunk import decode_cursor
from domain.knowledge.enums import DocumentVisibility
from infrastructure.database.models import ChunkModel
from infrastructure.repositories.chunk.chunk_mapping import to_chunk_search_result
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import (
    SQLAlchemyChunkRepository,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["next", "prev"])
async def test_cursor_listing_preserves_metadata_and_page_boundaries(direction: str) -> None:
    rows = [
        ChunkModel(
            id=index + 10,
            document_id=7,
            filename="rules.pdf",
            content=f"Rule {index}",
            chunk_index=index,
            visibility=DocumentVisibility.INTERNAL_PRIVATE.value,
            doc_domain="general",
            owner_id=12,
            group_id=3,
            edited_at=datetime(2026, 1, 1, tzinfo=UTC),
            edited_by=12,
            manual=True,
            creation_date=datetime(2025, 1, 1, tzinfo=UTC),
            content_hash=f"hash-{index}",
            section="Applications",
            heading="Required documents",
            heading_level=2,
            content_type="text",
            doc_title="Rules",
            doc_type="decree",
            act_version_id=9,
            effective_from=date(2025, 1, 1),
            effective_to=date(2025, 12, 31),
            is_current=False,
            context_metadata={"page": 4, "parent_units": [{"content": "Only legal entities."}]},
        )
        for index in range(3)
    ]
    fetched = rows if direction == "next" else list(reversed(rows))
    result = MagicMock()
    result.scalars.return_value.all.return_value = fetched
    session = MagicMock(spec=AsyncSession, execute=AsyncMock(return_value=result))
    page = await SQLAlchemyChunkRepository(session).list_for_document_cursor(
        7, limit=2, cursor=(5, 15), direction=direction
    )
    expected_rows = rows[:2] if direction == "next" else rows[1:]
    expected = [to_chunk_search_result(row) for row in expected_rows]
    assert page.items == expected
    assert [to_chunk_dto(item) for item in page.items] == [to_chunk_dto(item) for item in expected]
    assert page.next_cursor is not None
    assert page.prev_cursor is not None
    assert decode_cursor(page.next_cursor) == (expected[-1].chunk_index, expected[-1].chunk_id)
    assert decode_cursor(page.prev_cursor) == (expected[0].chunk_index, expected[0].chunk_id)
