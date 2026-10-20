"""Source identity migration refuses ambiguous links and separates ingestion slots."""

from collections.abc import Callable

from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
from sqlalchemy import Connection, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, async_sessionmaker

from domain.entities.document import Document
from infrastructure.database.migrations.versions import s9i0j1k2l3m4_ingestion_source_identity as migration
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def apply_migration(connection: Connection, action: Callable[[], None]) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        action()


async def legacy_document(
    connection: AsyncConnection, filename: str = "a.txt", source: str = "", source_type: str = "file"
) -> int | None:
    result: int | None = await connection.scalar(
        text(
            "INSERT INTO documents "
            "(filename, source_path, visibility, status, doc_domain, source_type, "
            "creation_date) "
            "VALUES (:filename, :source, 'internal_public', 'done', 'general', "
            ":source_type, now()) RETURNING id"
        ),
        {"filename": filename, "source": source, "source_type": source_type},
    )
    return result


async def legacy_registry(
    connection: AsyncConnection, filename: str = "a.txt", source: str = "s3://bucket/docs/sub/a.txt"
) -> None:
    await connection.execute(
        text(
            "INSERT INTO ingestion_registry (filename, source, creation_date) "
            "VALUES (:filename, :source, now())"
        ),
        {"filename": filename, "source": source},
    )


async def test_migration_transfers_confirmed_links_and_retains_ui_documents(pg_engine: AsyncEngine) -> None:
    async with pg_engine.begin() as connection:
        await connection.run_sync(apply_migration, migration.downgrade)
        document_id = await legacy_document(connection)
        upload_id = await legacy_document(connection, "upload.txt", "uploads/upload.txt")
        manual_id = await legacy_document(connection, "manual.txt", source_type="manual")
        await legacy_registry(connection)
        await legacy_registry(connection, "upload.txt", "s3://bucket/docs/upload.txt")
        await legacy_registry(connection, "manual.txt", "s3://bucket/docs/manual.txt")
        await connection.run_sync(apply_migration, migration.upgrade)
        result = await connection.execute(text("SELECT id, ingestion_key FROM documents"))
        rows = {row.id: row.ingestion_key for row in result}
        assert rows == {document_id: "docs/sub/a.txt", upload_id: None, manual_id: None}
        assert set(
            (await connection.execute(text("SELECT ingestion_key FROM ingestion_registry"))).scalars()
        ) == {"docs/sub/a.txt", "docs/upload.txt", "docs/manual.txt"}
        # A second directory and a UI upload may share the basename and assignment.
        other = await legacy_document(connection, "a.txt", "docs/other/a.txt")
        # The row starts in the UI slot, then moves into its ingestion slot.
        await connection.execute(
            text("UPDATE documents SET ingestion_key = 'docs/other/a.txt' WHERE id = :id"), {"id": other}
        )
        await legacy_document(connection, "a.txt", "uploads/a.txt")
        async with connection.begin_nested() as savepoint:
            with pytest.raises(IntegrityError):
                await connection.execute(
                    text("UPDATE documents SET ingestion_key = 'docs/sub/a.txt' WHERE id = :id"),
                    {"id": other},
                )
            await savepoint.rollback()
        async with connection.begin_nested() as savepoint:
            with pytest.raises(IntegrityError):
                await legacy_document(connection, "a.txt", "uploads/duplicate.txt")
            await savepoint.rollback()


async def test_ambiguous_legacy_link_aborts_without_partial_schema_or_data_changes(
    pg_engine: AsyncEngine,
) -> None:
    async with pg_engine.begin() as connection:
        await connection.run_sync(apply_migration, migration.downgrade)
        document_id = await legacy_document(connection, source="s3://bucket/docs/other/a.txt")
        await legacy_registry(connection)
        # Conflicting explicit source evidence for the same document.
        await connection.execute(
            text(
                "INSERT INTO vector_store_outbox "
                "(operation, aggregate_type, aggregate_id, payload, status, attempts, max_attempts, "
                "next_attempt_at, creation_date) "
                "VALUES ('set_document_id', 'document', :id, CAST(:payload AS JSON), "
                "'pending', 0, 8, now(), now())"
            ).bindparams(payload='{"source": "s3://bucket/docs/sub/a.txt"}'),
            {"id": document_id},
        )
        async with connection.begin_nested() as savepoint:
            with pytest.raises(ValueError, match="Ambiguous legacy ingestion document"):
                await connection.run_sync(apply_migration, migration.upgrade)
            await savepoint.rollback()
        assert await connection.scalar(text("SELECT count(*) FROM documents")) == 1
        assert (
            await connection.scalar(
                text(
                    "SELECT count(*) FROM information_schema.columns WHERE table_schema = current_schema() "
                    "AND column_name = 'ingestion_key' AND table_name IN ('documents', 'ingestion_registry')"
                )
            )
            == 0
        )
        assert (
            await connection.scalar(text("SELECT source FROM ingestion_registry"))
            == "s3://bucket/docs/sub/a.txt"
        )


async def test_sql_prefix_is_literal_and_case_sensitive(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    keys = ["docs/s_b/a.txt", "docs/sub/a.txt", "docs/s_bmarine/a.txt", "docs/S_b/a.txt", "docs/s%b/a.txt"]
    async with sessions.begin() as session:
        repo = SQLAlchemyDocumentRepository(session)

        for key in keys:
            await repo.save(Document(filename="a.txt", ingestion_key=key))

        assert [doc.ingestion_key for doc in await repo.list_ingestion_documents("docs/s_b/")] == [keys[0]]
        assert [doc.ingestion_key for doc in await repo.list_ingestion_documents("docs/s%b/")] == [keys[4]]


async def test_migration_keeps_multiple_assignments_for_the_same_source(pg_engine: AsyncEngine) -> None:
    async with pg_engine.begin() as connection:
        await connection.run_sync(apply_migration, migration.downgrade)
        owner_id = await connection.scalar(
            text(
                "INSERT INTO users (email, hashed_password, role, kind, is_active, creation_date) "
                "VALUES ('client@example.org', 'test', 'user', 'client', true, now()) RETURNING id"
            )
        )
        group_id = await connection.scalar(
            text("INSERT INTO groups (name, creation_date) VALUES ('assigned', now()) RETURNING id")
        )
        private_id = await legacy_document(connection)
        await connection.execute(
            text("UPDATE documents SET owner_id = :owner, visibility = 'client_private' WHERE id = :id"),
            {"owner": owner_id, "id": private_id},
        )
        group_doc_id = await legacy_document(connection)
        await connection.execute(
            text("UPDATE documents SET group_id = :group_id, visibility = 'internal_group' WHERE id = :id"),
            {"group_id": group_id, "id": group_doc_id},
        )
        await legacy_registry(connection)
        await connection.run_sync(apply_migration, migration.upgrade)
        result = await connection.execute(
            text("SELECT id, owner_id, group_id, visibility, ingestion_key FROM documents ORDER BY id")
        )
        assert [tuple(row) for row in result] == [
            (private_id, owner_id, None, "client_private", "docs/sub/a.txt"),
            (group_doc_id, None, group_id, "internal_group", "docs/sub/a.txt"),
        ]
