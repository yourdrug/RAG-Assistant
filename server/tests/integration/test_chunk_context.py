"""PostgreSQL round trips and migration contracts for chunk interpretation metadata."""

from importlib import import_module

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.search_mode import SearchMode
from domain.value_objects.user_context import UserContext
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.database.models import DocumentModel, UserModel
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_context_round_trip_and_unauthorized_search(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    context = {
        "parent_units": [{"content": "Only new applications are eligible."}],
        "pages": [2, 3],
        "page_start": 2,
        "page_end": 3,
        "subpoint_num_number": "21.2",
    }
    async with sessions.begin() as session:
        users = [
            UserModel(
                email=f"user-{i}@example.org",
                hashed_password="test",
                role=UserRole.USER.value,
                kind=UserKind.INTERNAL.value,
            )
            for i in range(2)
        ]
        session.add_all(users)
        await session.flush()
        doc = DocumentModel(
            filename="rules.pdf",
            visibility=DocumentVisibility.INTERNAL_PRIVATE.value,
            owner_id=users[0].id,
            doc_domain=DocDomain.DECREE.value,
        )
        session.add(doc)
        await session.flush()
        ids = await SQLAlchemyChunkRepository(session).bulk_insert(
            doc.id,
            doc.filename,
            doc.visibility,
            ["21.2. Submit the passport."],
            owner_id=users[0].id,
            doc_domain=DocDomain.DECREE.value,
            content_hashes=["context-hash"],
            context_metadata=[context],
        )
    allowed = UserContext(users[0].id, UserKind.INTERNAL, UserRole.USER)
    denied = UserContext(users[1].id, UserKind.INTERNAL, UserRole.USER)
    async with sessions() as session:
        repository = SQLAlchemyChunkRepository(session)
        saved = await repository.get_by_id(ids[0])
        assert saved.context_metadata == context
        matches = await repository.search_substring("passport", allowed, mode=SearchMode.ICONTAINS.value)
        assert len(matches) == 1 and matches[0].context_metadata == context
        neighbors = await repository.get_neighbors(doc.id, 0, user=allowed)
        assert len(neighbors) == 1 and neighbors[0].context_metadata == context
        assert await repository.search_substring("passport", denied, mode=SearchMode.ICONTAINS.value) == []
        assert await repository.get_neighbors(doc.id, 0, user=denied) == []


async def test_context_migration_preserves_legacy_rows_and_enforces_objects(pg_engine):
    migration = import_module(
        "infrastructure.database.migrations.versions.f9a0b1c2d3e4_add_chunk_context_metadata"
    )

    def run_migration(connection, operation):
        with Operations.context(MigrationContext.configure(connection)):
            operation()

    async with pg_engine.begin() as conn:
        # Recreate the prior schema only inside the disposable test schema.
        await conn.run_sync(lambda connection: run_migration(connection, migration.downgrade))
        doc_id = (
            await conn.execute(
                text(
                    "INSERT INTO documents "
                    "(filename, visibility, status, source_path, doc_domain, creation_date) "
                    "VALUES ('legacy.pdf', :visibility, :status, '', :domain, CURRENT_TIMESTAMP) RETURNING id"
                ),
                {
                    "visibility": DocumentVisibility.INTERNAL_PUBLIC.value,
                    "domain": DocDomain.GENERAL.value,
                    "status": DocumentStatus.DONE.value,
                },
            )
        ).scalar_one()
        chunk_id = (
            await conn.execute(
                text(
                    "INSERT INTO chunks "
                    "(document_id, chunk_index, content, filename, visibility, doc_domain, creation_date) "
                    "VALUES (:doc_id, 0, 'Legacy content', 'legacy.pdf', :visibility, :domain, "
                    "CURRENT_TIMESTAMP) "
                    "RETURNING id"
                ),
                {
                    "doc_id": doc_id,
                    "visibility": DocumentVisibility.INTERNAL_PUBLIC.value,
                    "domain": DocDomain.GENERAL.value,
                },
            )
        ).scalar_one()
        await conn.run_sync(lambda connection: run_migration(connection, migration.upgrade))
        row = (
            await conn.execute(
                text("SELECT content, context_metadata FROM chunks WHERE id = :id"), {"id": chunk_id}
            )
        ).one()
        assert row == ("Legacy content", {})
        with pytest.raises(IntegrityError):
            async with conn.begin_nested():
                await conn.execute(
                    text("UPDATE chunks SET context_metadata = '[]'::jsonb WHERE id = :id"), {"id": chunk_id}
                )
        with pytest.raises(IntegrityError):
            async with conn.begin_nested():
                await conn.execute(
                    text("UPDATE chunks SET context_metadata = NULL WHERE id = :id"), {"id": chunk_id}
                )
        await conn.run_sync(lambda connection: run_migration(connection, migration.downgrade))
        assert (
            await conn.execute(text("SELECT content FROM chunks WHERE id = :id"), {"id": chunk_id})
        ).scalar_one() == ("Legacy content")
