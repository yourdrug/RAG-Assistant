"""PostgreSQL ingestion locks preserve concurrent assignment changes."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from application.ingestion import DocumentSyncService
from domain.entities.document import Document
from domain.entities.raw_document import RawDocument
from domain.identity.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.database.models import UserModel
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
from infrastructure.uow_factory import UnitOfWorkFactory

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_reindex_reads_concurrent_assignment_after_acquiring_row_lock(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        client = UserModel(
            email="client@example.org",
            hashed_password="test",
            role=UserRole.USER.value,
            kind=UserKind.CLIENT.value,
        )
        session.add(client)
        await session.flush()
        client_id = client.id
        doc = await SQLAlchemyDocumentRepository(session).save(
            Document(filename="report.md", ingestion_key="docs/report.md")
        )
        doc_id = doc.id
        await SQLAlchemyChunkRepository(session).bulk_insert(
            doc_id, "report.md", DocumentVisibility.INTERNAL_PUBLIC.value, ["old public text"]
        )

    class SessionDatabase:
        def get_session(self, master=False):
            return sessions()

    real_factory = UnitOfWorkFactory(SessionDatabase())
    attempted = asyncio.Event()
    locked = asyncio.Event()

    class GatedRepository(SQLAlchemyDocumentRepository):
        async def get_by_id(self, document_id, *, for_update=False):
            if for_update:
                attempted.set()
            doc = await super().get_by_id(document_id, for_update=for_update)
            if for_update:
                locked.set()
            return doc

    class GatedFactory:
        @asynccontextmanager
        async def create(self, master=False):
            async with real_factory.create(master=master) as uow:
                uow.documents = GatedRepository(uow._session)
                yield uow

    # Model the UI changing an assignment while reset has already parsed the file.
    async with sessions.begin() as session:
        await SQLAlchemyDocumentRepository(session).set_acl(
            doc_id, DocumentVisibility.CLIENT_PRIVATE, client_id, None
        )
        indexing = asyncio.create_task(
            DocumentSyncService(GatedFactory()).sync_documents_to_db(
                {"docs/report.md": {"source": "s3://bucket/docs/report.md"}},
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
    async with real_factory.create(master=True) as uow:
        doc = await uow.documents.get_by_id(doc_id)
        chunks, total = await uow.chunks.list_for_document(doc_id)
        assert doc.owner_id == client_id
        assert doc.visibility == DocumentVisibility.CLIENT_PRIVATE
        assert total == 1
        assert chunks[0].content == "new private text"
        assert chunks[0].owner_id == client_id
        assert chunks[0].visibility == DocumentVisibility.CLIENT_PRIVATE.value
        entries = await uow.vector_outbox.claim_batch("test", limit=10)
        point = next(entry.payload["points"][0] for entry in entries if "points" in entry.payload)
        assert point["metadata"]["owner_id"] == client_id
        assert point["metadata"]["document_id"] == doc_id


async def test_ingestion_session_lock_serializes_without_open_transaction_and_releases_on_cancel(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)

    class SessionDatabase:
        def get_session(self, master=False):
            return sessions()

    factory = UnitOfWorkFactory(SessionDatabase())
    first_locked = asyncio.Event()
    second_locked = asyncio.Event()
    release = asyncio.Event()

    async def hold(event):
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
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await asyncio.wait_for(second_locked.wait(), timeout=5)
    finally:
        release.set()
        await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)
    async with factory.ingestion_lock():
        pass
