"""SEC-001: concurrent ingestion must commit document and chunk ACL together."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from application.ingestion import DocumentSyncService
from domain.entities.document import Document
from domain.entities.raw_document import RawDocument
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.database.models import UserModel
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
from infrastructure.uow_factory import UnitOfWorkFactory

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_concurrent_reindex_serializes_acl_and_content_replacement(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        clients = [
            UserModel(
                email=email,
                hashed_password="test",
                role=UserRole.USER.value,
                kind=UserKind.CLIENT.value,
            )
            for email in ("first@example.org", "second@example.org")
        ]
        session.add_all(clients)
        await session.flush()
        first_client, second_client = [client.id for client in clients]
        doc = await SQLAlchemyDocumentRepository(session).save(Document(filename="report.md"))
        doc_id = doc.id
        await SQLAlchemyChunkRepository(session).bulk_insert(
            doc_id, "report.md", DocumentVisibility.INTERNAL_PUBLIC.value, ["old public text"]
        )

    first_locked = asyncio.Event()
    release_first = asyncio.Event()
    second_attempted = asyncio.Event()
    second_locked = asyncio.Event()

    class SessionDatabase:
        def get_session(self, master=False):
            return sessions()

    real_factory = UnitOfWorkFactory(SessionDatabase())

    class GatedDocumentRepository(SQLAlchemyDocumentRepository):
        async def set_acl(self, document_id, visibility, owner_id, group_id):
            if owner_id == second_client:
                second_attempted.set()
            await super().set_acl(document_id, visibility, owner_id, group_id)
            if owner_id == first_client:
                first_locked.set()
                await release_first.wait()
            else:
                second_locked.set()

    class GatedFactory:
        @asynccontextmanager
        async def create(self, master=False):
            async with real_factory.create(master=master) as uow:
                uow.documents = GatedDocumentRepository(uow._session)
                yield uow

    sync = DocumentSyncService(GatedFactory())
    source = "s3://bucket/docs/report.md"

    async def reindex(client_id, content):
        await sync.sync_documents_to_db(
            {"report.md": {"source": source}},
            {},
            [RawDocument(page_content=content, metadata={"source": source})],
            visibility=DocumentVisibility.CLIENT_PRIVATE,
            client_id=client_id,
        )

    tasks = [asyncio.create_task(reindex(first_client, "first client's text"))]
    try:
        await asyncio.wait_for(first_locked.wait(), timeout=10)
        tasks.append(asyncio.create_task(reindex(second_client, "second client's text")))
        await asyncio.wait_for(second_attempted.wait(), timeout=10)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(second_locked.wait(), timeout=0.1)

        # A separate reader sees the original document and content until commit.
        async with sessions() as session:
            before = await SQLAlchemyDocumentRepository(session).get_by_id(doc_id)
            chunks, total = await SQLAlchemyChunkRepository(session).list_for_document(doc_id)
            assert before.visibility == DocumentVisibility.INTERNAL_PUBLIC
            assert before.owner_id is None
            assert total == 1
            assert chunks[0].visibility == DocumentVisibility.INTERNAL_PUBLIC.value
            assert chunks[0].content == "old public text"
    finally:
        release_first.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)

    async with sessions() as session:
        after = await SQLAlchemyDocumentRepository(session).get_by_id(doc_id)
        chunks, total = await SQLAlchemyChunkRepository(session).list_for_document(doc_id)
        assert after.visibility == DocumentVisibility.CLIENT_PRIVATE
        assert after.owner_id == second_client
        assert after.group_id is None
        assert total == 1
        assert chunks[0].content == "second client's text"
        assert chunks[0].visibility == DocumentVisibility.CLIENT_PRIVATE.value
        assert chunks[0].owner_id == second_client
        assert chunks[0].group_id is None
