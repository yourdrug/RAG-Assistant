"""PostgreSQL ingestion locks preserve concurrent assignment changes."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tests.integration.uow import SessionDatabase

from application.ingestion import DocumentSyncService
from domain.identity.roles import UserKind, UserRole
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.chunk import ChunkInsert
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentVisibility
from infrastructure.database.database import DatabaseManager
from infrastructure.database.models import UserModel
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
from infrastructure.uow_factory import UnitOfWorkFactory


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine

    from application.uow.unit_of_work import UnitOfWork


TEST_HASH = "test"


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_reindex_reads_concurrent_assignment_after_acquiring_row_lock(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    doc_id, client_id = await seed_public_document(sessions)
    database = SessionDatabase(sessions)
    real_factory = UnitOfWorkFactory(database)
    attempted = asyncio.Event()
    locked = asyncio.Event()

    class GatedRepository(SQLAlchemyDocumentRepository):
        async def get_by_id(self, document_id: int, *, for_update: bool = False) -> Document | None:
            if for_update:
                attempted.set()

            doc = await super().get_by_id(document_id, for_update=for_update)
            assert doc is not None

            if for_update:
                locked.set()

            return doc

    class GatedFactory(UnitOfWorkFactory):
        @asynccontextmanager
        async def create(self, *, master: bool = False) -> AsyncIterator[UnitOfWork]:
            async with real_factory.create(master=master) as uow:
                assert isinstance(uow._session, AsyncSession)
                uow.documents = GatedRepository(uow._session)
                yield uow

    # Model the UI changing an assignment while reset has already parsed the file.
    async with sessions.begin() as session:
        await SQLAlchemyDocumentRepository(session).set_acl(
            doc_id, DocumentVisibility.CLIENT_PRIVATE, client_id, None
        )
        indexing = asyncio.create_task(
            DocumentSyncService(GatedFactory(database)).sync_documents_to_db(
                {
                    "docs/report.md": {
                        "source": "s3://bucket/docs/report.md",
                        "chars": 16,
                        "chunks": 1,
                        "hash": "test",
                    }
                },
                {},
                [
                    RawDocument(
                        page_content="new private text", metadata={"source": "s3://bucket/docs/report.md"}
                    )
                ],
            )
        )
        await asyncio.wait_for(attempted.wait(), timeout=5)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(locked.wait(), timeout=0.1)
    await asyncio.wait_for(indexing, timeout=5)
    await assert_private_reindex(real_factory, doc_id, client_id)


async def test_ingestion_session_lock_serializes_without_open_transaction_and_releases_on_cancel(
    pg_engine: AsyncEngine,
) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)

    database = DatabaseManager()
    database.create_master_connection_engines(pg_engine.url.set(drivername="postgresql+psycopg"), {})
    factory = UnitOfWorkFactory(database)
    first_locked = asyncio.Event()
    second_locked = asyncio.Event()
    release = asyncio.Event()

    async def hold(event: asyncio.Event) -> None:
        async with factory.ingestion_lock():
            event.set()
            await release.wait()

    first = asyncio.create_task(hold(first_locked))
    second = None
    try:
        await asyncio.wait_for(first_locked.wait(), timeout=5)
        second = asyncio.create_task(hold(second_locked))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(second_locked.wait(), timeout=0.1)
        await assert_advisory_connection_idle(sessions)
        await cancel_and_wait_for_lock(first, second_locked)
        release.set()
        await second
        async with factory.ingestion_lock():
            pass
    finally:
        release.set()
        await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)
        await database.disconnect()


async def seed_public_document(sessions: async_sessionmaker[AsyncSession]) -> tuple[int, int]:
    async with sessions.begin() as session:
        client = UserModel(
            email="client@example.org",
            hashed_password=TEST_HASH,
            role=UserRole.USER.value,
            kind=UserKind.CLIENT.value,
        )
        session.add(client)
        await session.flush()
        client_id = client.id
        doc = await SQLAlchemyDocumentRepository(session).save(
            Document(filename="report.md", ingestion_key="docs/report.md")
        )
        assert doc.id is not None
        doc_id = doc.id
        await SQLAlchemyChunkRepository(session).bulk_insert(
            doc_id,
            "report.md",
            DocumentVisibility.INTERNAL_PUBLIC.value,
            [ChunkInsert(content="old public text")],
        )

    return doc_id, client_id


async def assert_private_reindex(factory: UnitOfWorkFactory, doc_id: int, client_id: int) -> None:
    async with factory.create(master=True) as uow:
        loaded = await uow.documents.get_by_id(doc_id)
        assert loaded is not None
        chunks, total = await uow.chunks.list_for_document(doc_id)
        assert loaded.owner_id == client_id
        assert loaded.visibility == DocumentVisibility.CLIENT_PRIVATE
        assert total == 1
        assert chunks[0].content == "new private text"
        assert chunks[0].owner_id == client_id
        assert chunks[0].visibility == DocumentVisibility.CLIENT_PRIVATE.value
        entries = await uow.vector_outbox.claim_batch("test", limit=10)
        point = next(entry.payload["points"][0] for entry in entries if "points" in entry.payload)
        assert point["metadata"]["owner_id"] == client_id
        assert point["metadata"]["document_id"] == doc_id


async def assert_advisory_connection_idle(sessions: async_sessionmaker[AsyncSession]) -> None:
    async with sessions() as session:
        states = (
            (
                await session.execute(
                    text(
                        "SELECT state FROM pg_stat_activity WHERE pid IN "
                        "(SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND granted)"
                    )
                )
            )
            .scalars()
            .all()
        )
        assert states == ["idle"]


async def cancel_and_wait_for_lock(first: asyncio.Task[None], second_locked: asyncio.Event) -> None:
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(second_locked.wait(), timeout=5)
