"""PostgreSQL round trips and migration contracts for chunk interpretation metadata."""

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from domain.knowledge.chunk import ChunkInsert
from domain.knowledge.enums import DocDomain
from domain.knowledge.enums import DocumentStatus
from domain.identity.roles import UserKind, UserRole
from domain.knowledge.enums import SearchMode
from domain.identity.user_context import UserContext
from domain.knowledge.enums import DocumentVisibility
from infrastructure.database.models import DocumentModel, UserModel
from infrastructure.database.database import DatabaseManager
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository
from infrastructure.uow_factory import UnitOfWorkFactory

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_document_index_lock_serializes_factories_and_releases_after_error(pg_engine):
    database = DatabaseManager()
    database.async_master_connection_engine = create_async_engine(
        pg_engine.url, poolclass=NullPool, isolation_level="AUTOCOMMIT"
    )
    first, second = UnitOfWorkFactory(database), UnitOfWorkFactory(database)
    acquired = asyncio.Event()

    async def competing_upload():
        async with second.document_index_lock(1):
            acquired.set()

    async def failing_upload(group):
        async with first.document_index_lock(1):
            group.create_task(competing_upload())
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(acquired.wait(), timeout=0.1)
            async with second.document_index_lock(2):
                pass
            message = "upload failed"
            raise ValueError(message)

    try:
        async with asyncio.timeout(5), asyncio.TaskGroup() as group:
            with pytest.raises(ValueError, match="upload failed"):
                await failing_upload(group)
            await acquired.wait()
    finally:
        await database.async_master_connection_engine.dispose()


async def test_batch_chunk_read_uses_committed_content_and_document_scope(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        documents = [
            DocumentModel(filename=f"outbox-{index}.txt", visibility=DocumentVisibility.INTERNAL_PUBLIC.value)
            for index in range(2)
        ]
        session.add_all(documents)
        await session.flush()
        repository = SQLAlchemyChunkRepository(session)
        ids = []

        for document in documents:
            ids.extend(
                await repository.bulk_insert(
                    document.id,
                    document.filename,
                    DocumentVisibility.INTERNAL_PUBLIC.value,
                    [ChunkInsert(content="old text")],
                )
            )

        document_id = documents[0].id

    async with sessions.begin() as session:
        await session.execute(text("UPDATE chunks SET content = 'new text' WHERE id = :id"), {"id": ids[0]})

    async with sessions() as session:
        repository = SQLAlchemyChunkRepository(session)
        chunks = await repository.get_by_ids(document_id, [*ids, 999999])
        assert [(chunk.chunk_id, chunk.content) for chunk in chunks] == [(ids[0], "new text")]
        assert await repository.get_by_ids(document_id, []) == []


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
            [
                ChunkInsert(
                    content="21.2. Submit the passport.",
                    content_hash="context-hash",
                    context_metadata=context,
                )
            ],
            owner_id=users[0].id,
            doc_domain=DocDomain.DECREE.value,
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


async def test_context_default_and_object_constraint(pg_engine):

    async with pg_engine.begin() as conn:
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
        assert (
            await conn.execute(text("SELECT content FROM chunks WHERE id = :id"), {"id": chunk_id})
        ).scalar_one() == ("Legacy content")
