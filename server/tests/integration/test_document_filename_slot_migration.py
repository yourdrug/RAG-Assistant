"""Document slot migration preserves duplicates and enforces indexing on PostgreSQL."""

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

from domain.knowledge.enums import DocDomain
from domain.knowledge.enums import DocumentStatus
from domain.knowledge.enums import DocumentVisibility
from infrastructure.database.migrations.versions import (
    n4e5f6g7h8i9_include_indexing_in_document_slot as migration,
)

pytestmark = pytest.mark.integration


def apply_migration(connection, action):
    with Operations.context(MigrationContext.configure(connection)):
        action()


async def insert_document(connection, status):
    return await connection.scalar(
        text(
            "INSERT INTO documents (filename, source_path, visibility, status, doc_domain, creation_date) "
            "VALUES ('report.pdf', 'report.pdf', :visibility, :status, :domain, CURRENT_TIMESTAMP) "
            "RETURNING id"
        ),
        {
            "visibility": DocumentVisibility.INTERNAL_PUBLIC.value,
            "status": status.value,
            "domain": DocDomain.GENERAL.value,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_status", list(DocumentStatus))
async def test_duplicate_preflight_retains_rows_and_old_index(pg_engine, existing_status):
    async with pg_engine.begin() as connection:
        await connection.run_sync(apply_migration, migration.downgrade)
        await insert_document(connection, existing_status)
        await insert_document(connection, DocumentStatus.INDEXING)
        old_index = await connection.scalar(
            text(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() "
                "AND indexname = 'ux_documents_active_slot'"
            )
        )
        async with connection.begin_nested() as savepoint:
            with pytest.raises(DBAPIError, match="duplicate document IDs"):
                await connection.run_sync(apply_migration, migration.upgrade)
            await savepoint.rollback()
        assert await connection.scalar(text("SELECT count(*) FROM documents")) == 2
        assert (
            await connection.scalar(
                text(
                    "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() "
                    "AND indexname = 'ux_documents_active_slot'"
                )
            )
            == old_index
        )


@pytest.mark.asyncio
async def test_valid_migration_reserves_indexing_slot_and_downgrades(pg_engine):
    async with pg_engine.begin() as connection:
        await connection.run_sync(apply_migration, migration.downgrade)
        document_id = await insert_document(connection, DocumentStatus.INDEXING)
        await connection.run_sync(apply_migration, migration.upgrade)
        assert await connection.scalar(text("SELECT id FROM documents")) == document_id
        async with connection.begin_nested() as savepoint:
            with pytest.raises(IntegrityError):
                await insert_document(connection, DocumentStatus.PENDING)
            await savepoint.rollback()
        await connection.run_sync(apply_migration, migration.downgrade)
        await insert_document(connection, DocumentStatus.PENDING)
        assert await connection.scalar(text("SELECT count(*) FROM documents")) == 2
