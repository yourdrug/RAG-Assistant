"""Real PostgreSQL tests for concurrent claim, notification and duplicate uploads."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tests.integration.uow import SessionDatabase

from application.documents import DocumentCommandService
from application.documents.delete import DocumentDeleteCommand
from application.documents.rename import DocumentRenameCommand
from application.documents.upload import DocumentUploadCommand
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.document import Document
from domain.knowledge.enums import GENERAL_DOMAIN_KEY, DocumentVisibility
from domain.knowledge.index_sync import VectorOutboxEntry
from domain.runtime.jobs import BackgroundJob, BackgroundJobStatus, JobType
from domain.shared.exceptions import BusinessRuleViolationError
from infrastructure.database.models import BackgroundJobModel, DocumentModel, UserModel
from infrastructure.repositories.config.sqlalchemy_vector_outbox_repository import (
    SQLAlchemyVectorOutboxRepository,
)
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
from infrastructure.repositories.misc.sqlalchemy_background_job_repository import (
    SQLAlchemyBackgroundJobRepository,
)
from infrastructure.uow_factory import UnitOfWorkFactory


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine

    from application.uow.unit_of_work import UnitOfWork


TEST_HASH = "test"


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_two_claimers_skip_locks_and_rollback_restores_claim(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyVectorOutboxRepository(session)
        entries = [await repo.enqueue(VectorOutboxEntry(aggregate_id=i)) for i in (11, 12)]
    barrier = asyncio.Barrier(2)
    claimed = asyncio.Event()

    async def first() -> list[VectorOutboxEntry]:
        async with sessions() as session:
            result = await SQLAlchemyVectorOutboxRepository(session).claim_batch("one", limit=1)
            await barrier.wait()
            await claimed.wait()
            await session.rollback()
            return result

    async def second() -> list[VectorOutboxEntry]:
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


async def test_outbox_notification_arrives_only_after_commit(pg_engine: AsyncEngine) -> None:
    listener = await asyncpg.connect(
        os.environ["TEST_POSTGRES_URL"].replace("postgresql+asyncpg://", "postgresql://")
    )
    notified = asyncio.Event()
    await listener.add_listener("vector_outbox_ready", lambda *_args: notified.set())
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


async def test_three_versions_preserve_original_anchor(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyDocumentRepository(session)
        first = await repo.save(Document(filename="v1.pdf", doc_domain=GENERAL_DOMAIN_KEY))
        assert first.id is not None
        second = await repo.save(
            Document(filename="v2.pdf", doc_domain=GENERAL_DOMAIN_KEY, version_group_id=first.id)
        )
        assert second.id is not None
    async with sessions.begin() as session:
        repo = SQLAlchemyDocumentRepository(session)
        loaded = await repo.get_by_id(second.id)
        assert loaded is not None
        third = await repo.save(
            Document(
                filename="v3.pdf", doc_domain=GENERAL_DOMAIN_KEY, version_group_id=loaded.version_group_id
            )
        )
        assert third.id is not None
    async with sessions() as session:
        loaded_third = await SQLAlchemyDocumentRepository(session).get_by_id(third.id)
        assert loaded_third is not None
        assert loaded_third.version_group_id == first.id


async def test_concurrent_duplicate_upload_hits_real_unique_constraint(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        user = UserModel(email="race@example.org", hashed_password=TEST_HASH, role="user", kind="internal")
        session.add(user)
        await session.flush()
        user_id = user.id
    database = SessionDatabase(sessions)
    real_factory = UnitOfWorkFactory(database)
    barrier = asyncio.Barrier(2)

    class RacingRepository(SQLAlchemyDocumentRepository):
        async def find_active_slot(self, *args: Any, **kwargs: Any) -> Document | None:
            result = await super().find_active_slot(*args, **kwargs)
            await barrier.wait()
            return result

    class RacingFactory(UnitOfWorkFactory):
        @asynccontextmanager
        async def create(self, *, master: bool = False) -> AsyncIterator[UnitOfWork]:
            async with real_factory.create(master=master) as uow:
                assert isinstance(uow._session, AsyncSession)
                uow.documents = RacingRepository(uow._session)
                yield uow

    context = UserContext(user_id=user_id, user_kind=UserKind.INTERNAL, user_role=UserRole.USER, group_ids=())
    context_factory = MagicMock()
    context_factory.build = AsyncMock(return_value=context)
    storage = MagicMock(supported_extensions=(".pdf",))
    storage.upload_file = AsyncMock()
    storage.delete_file = AsyncMock()
    factory = RacingFactory(database)
    command = DocumentCommandService(
        DocumentUploadCommand(factory, storage, context_factory),
        DocumentDeleteCommand(factory, storage, context_factory),
        DocumentRenameCommand(factory, storage, context_factory),
    )

    async def upload() -> object:
        return await command.upload(
            filename="race.pdf",
            file_data=b"pdf",
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
            group_id=None,
            user_id=user_id,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.USER,
        )

    results = await asyncio.wait_for(asyncio.gather(upload(), upload(), return_exceptions=True), timeout=10)
    assert sum(not isinstance(r, BaseException) for r in results) == 1
    assert sum(isinstance(r, BusinessRuleViolationError) for r in results) == 1
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(DocumentModel)) == 1
        job = (await session.execute(select(BackgroundJobModel))).scalar_one()
        assert job.dispatch_payload is not None
        assert job.dispatch_payload["document_id"] == job.related_id
        assert job.dispatch_payload["cost_principal"] == f"user:{user_id}"


async def test_waiting_job_heartbeat_prevents_false_stale_failure(pg_engine: AsyncEngine) -> None:

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


async def test_document_dispatch_claims_skip_locked_and_retry_after_rollback(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBackgroundJobRepository(session)
        jobs = [
            await repo.create(
                BackgroundJob(
                    job_type=JobType.DOCUMENT_PROCESSING,
                    related_id=i,
                    dispatch_payload={"document_id": i},
                )
            )
            for i in (11, 12)
        ]
    async with sessions() as first, sessions() as second:
        one = await SQLAlchemyBackgroundJobRepository(first).pending_dispatch(limit=1)
        two = await SQLAlchemyBackgroundJobRepository(second).pending_dispatch(limit=1)
        assert [job.id for job in one] == [jobs[0].id]
        assert [job.id for job in two] == [jobs[1].id]
        await first.rollback()
        await second.rollback()
    async with sessions.begin() as session:
        repo = SQLAlchemyBackgroundJobRepository(session)
        assert len(await repo.pending_dispatch()) == 2
        assert jobs[0].id is not None
        await repo.mark_running(jobs[0].id)
    async with sessions.begin() as session:
        remaining = await SQLAlchemyBackgroundJobRepository(session).pending_dispatch()
        assert [job.id for job in remaining] == [jobs[1].id]


async def test_stale_pending_preserves_durable_dispatch_but_fails_legacy_jobs(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    old = datetime.now(tz=UTC) - timedelta(days=1)
    async with sessions.begin() as session:
        session.add_all(
            [
                BackgroundJobModel(
                    job_type=JobType.DOCUMENT_PROCESSING,
                    status="pending",
                    dispatch_payload={"document_id": 1},
                    creation_date=old,
                ),
                BackgroundJobModel(job_type=JobType.DOCUMENT_PROCESSING, status="pending", creation_date=old),
            ]
        )
    async with sessions.begin() as session:
        repo = SQLAlchemyBackgroundJobRepository(session)
        failed = await repo.fail_stale_pending(timeout_minutes=15)
        assert len(failed) == 1
        assert failed[0].dispatch_payload is None
        pending = await repo.pending_dispatch()
        assert len(pending) == 1
        assert pending[0].status == BackgroundJobStatus.PENDING
