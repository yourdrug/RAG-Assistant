"""Real PostgreSQL tests for concurrent claim, notification and duplicate uploads."""

import asyncio
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from application.services.document_command_service import DocumentCommandService
from domain.entities.document import Document
from domain.entities.vector_outbox_entry import VectorOutboxEntry
from domain.exceptions import BusinessRuleViolation
from domain.value_objects.user_context import UserContext
from infrastructure.database.models import DocumentModel, UserModel
from infrastructure.repositories.config.sqlalchemy_vector_outbox_repository import (
    SQLAlchemyVectorOutboxRepository,
)
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
from infrastructure.uow_factory import UnitOfWorkFactory

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_two_claimers_skip_locks_and_rollback_restores_claim(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyVectorOutboxRepository(session)
        entries = [await repo.enqueue(VectorOutboxEntry(aggregate_id=i)) for i in (11, 12)]
    barrier = asyncio.Barrier(2)
    claimed = asyncio.Event()

    async def first():
        async with sessions() as session:
            result = await SQLAlchemyVectorOutboxRepository(session).claim_batch("one", limit=1)
            await barrier.wait()
            await claimed.wait()
            await session.rollback()
            return result

    async def second():
        await barrier.wait()
        async with sessions.begin() as session:
            result = await SQLAlchemyVectorOutboxRepository(session).claim_batch("two", limit=1)
        claimed.set()
        return result

    a, b = await asyncio.wait_for(asyncio.gather(first(), second()), timeout=10)
    assert len(a) == len(b) == 1
    assert a[0].id != b[0].id
    assert {a[0].id, b[0].id} == {e.id for e in entries}
    async with sessions.begin() as session:
        reclaimed = await SQLAlchemyVectorOutboxRepository(session).claim_batch("three")
        assert [e.id for e in reclaimed] == [a[0].id]


async def test_outbox_notification_arrives_only_after_commit(pg_engine):
    listener = await asyncpg.connect(
        os.environ["TEST_POSTGRES_URL"].replace("postgresql+asyncpg://", "postgresql://")
    )
    notified = asyncio.Event()
    await listener.add_listener("vector_outbox_ready", lambda *args: notified.set())
    sessions = async_sessionmaker(pg_engine)
    try:
        async with sessions() as session:
            await SQLAlchemyVectorOutboxRepository(session).enqueue(VectorOutboxEntry(aggregate_id=123))
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(notified.wait(), timeout=0.1)
            await session.commit()
            await asyncio.wait_for(notified.wait(), timeout=3)
        notified.clear()
        async with sessions() as session:
            await SQLAlchemyVectorOutboxRepository(session).enqueue(VectorOutboxEntry(aggregate_id=124))
            await session.rollback()
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(notified.wait(), timeout=0.1)
    finally:
        await listener.close()


async def test_three_versions_preserve_original_anchor(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyDocumentRepository(session)
        first = await repo.save(Document(filename="v1.pdf", doc_domain="general"))
        second = await repo.save(Document(filename="v2.pdf", doc_domain="general", version_group_id=first.id))
    async with sessions.begin() as session:
        repo = SQLAlchemyDocumentRepository(session)
        loaded = await repo.get_by_id(second.id)
        third = await repo.save(
            Document(filename="v3.pdf", doc_domain="general", version_group_id=loaded.version_group_id)
        )
    async with sessions() as session:
        loaded_third = await SQLAlchemyDocumentRepository(session).get_by_id(third.id)
        assert loaded_third.version_group_id == first.id


async def test_concurrent_duplicate_upload_hits_real_unique_constraint(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        user = UserModel(email="race@example.org", hashed_password="test", role="user", kind="internal")
        session.add(user)
        await session.flush()
        user_id = user.id
    database = MagicMock()
    database.get_session.side_effect = lambda **kwargs: sessions()
    real_factory = UnitOfWorkFactory(database)
    barrier = asyncio.Barrier(2)

    class RacingRepository(SQLAlchemyDocumentRepository):
        async def find_active_slot(self, *args, **kwargs):
            result = await super().find_active_slot(*args, **kwargs)
            await barrier.wait()
            return result

    class RacingFactory:
        @asynccontextmanager
        async def create(self, master=False):
            async with real_factory.create(master=master) as uow:
                uow.documents = RacingRepository(uow._session)
                yield uow

    context = UserContext(user_id=user_id, user_kind="internal", user_role="user", group_ids=[])
    context_factory = MagicMock()
    context_factory.build = AsyncMock(return_value=context)
    storage = MagicMock(supported_extensions=(".pdf",))
    storage.upload_file = AsyncMock()
    storage.delete_file = AsyncMock()
    command = DocumentCommandService(
        RacingFactory(), MagicMock(), storage, MagicMock(), user_ctx_factory=context_factory
    )

    async def upload():
        return await command.upload(
            filename="race.pdf",
            file_data=b"pdf",
            visibility="internal_private",
            group_id=None,
            user_id=user_id,
            user_kind="internal",
            user_role="user",
        )

    results = await asyncio.wait_for(asyncio.gather(upload(), upload(), return_exceptions=True), timeout=10)
    assert sum(not isinstance(r, BaseException) for r in results) == 1
    assert sum(isinstance(r, BusinessRuleViolation) for r in results) == 1
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(DocumentModel)) == 1


async def test_waiting_job_heartbeat_prevents_false_stale_failure(pg_engine):
    from datetime import UTC, datetime, timedelta
    from infrastructure.database.models import BackgroundJobModel
    from infrastructure.repositories.misc.sqlalchemy_background_job_repository import (
        SQLAlchemyBackgroundJobRepository,
    )

    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        recent = BackgroundJobModel(
            job_type="document_processing",
            status="pending",
            creation_date=datetime.now(UTC) - timedelta(hours=1),
            heartbeat_at=datetime.now(UTC),
        )
        stale = BackgroundJobModel(
            job_type="document_processing",
            status="pending",
            creation_date=datetime.now(UTC) - timedelta(hours=1),
        )
        session.add_all([recent, stale])
        await session.flush()
        stale_id = stale.id
    async with sessions.begin() as session:
        failed = await SQLAlchemyBackgroundJobRepository(session).fail_stale_pending(timeout_minutes=15)
        assert [job.id for job in failed] == [stale_id]
