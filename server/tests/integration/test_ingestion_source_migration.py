"""Baseline ingestion slots keep full source keys separate from UI uploads."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from domain.knowledge.document import Document
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_sql_prefix_is_literal_and_case_sensitive(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    keys = ["docs/s_b/a.txt", "docs/sub/a.txt", "docs/s_bmarine/a.txt", "docs/S_b/a.txt", "docs/s%b/a.txt"]
    async with sessions.begin() as session:
        repo = SQLAlchemyDocumentRepository(session)

        for key in keys:
            await repo.save(Document(filename="a.txt", ingestion_key=key))

        assert [doc.ingestion_key for doc in await repo.list_ingestion_documents("docs/s_b/")] == [keys[0]]
        assert [doc.ingestion_key for doc in await repo.list_ingestion_documents("docs/s%b/")] == [keys[4]]


async def test_baseline_keeps_source_keys_separate_from_ui_slots(pg_engine: AsyncEngine) -> None:
    statement = text(
        "INSERT INTO documents (filename, visibility, ingestion_key) "
        "VALUES ('a.txt', 'internal_public', :key)"
    )
    async with pg_engine.begin() as connection:
        for key in ("docs/sub/a.txt", "docs/other/a.txt", None):
            await connection.execute(statement, {"key": key})

        for key in ("docs/sub/a.txt", None):
            with pytest.raises(IntegrityError):
                async with connection.begin_nested():
                    await connection.execute(statement, {"key": key})

        assert await connection.scalar(text("SELECT count(*) FROM documents")) == 3
