"""The baseline reserves document filename slots throughout indexing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from domain.knowledge.enums import DocumentStatus


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.mark.parametrize("existing_status", list(DocumentStatus))
async def test_baseline_reserves_filename_for_every_status(
    pg_engine: AsyncEngine, existing_status: DocumentStatus
) -> None:
    statement = text(
        "INSERT INTO documents (filename, visibility, status) "
        "VALUES ('report.pdf', 'internal_public', :status) RETURNING id"
    )
    async with pg_engine.begin() as connection:
        document_id = await connection.scalar(statement, {"status": existing_status.value})
        with pytest.raises(IntegrityError):
            async with connection.begin_nested():
                await connection.execute(statement, {"status": DocumentStatus.INDEXING.value})
        assert await connection.scalar(text("SELECT id FROM documents")) == document_id
