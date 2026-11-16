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
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from tests.integration.uow import SessionDatabase

from application.benchmark.dto import SweepCreateDTO
from application.benchmark.services import BenchmarkSweepService
from application.documents import DocumentCommandService
from application.documents.delete import DocumentDeleteCommand
from application.documents.rename import DocumentRenameCommand
from application.documents.upload import DocumentUploadCommand
from application.jobs.execution import ExecutionLeaseLostError, JobExecution, current_execution
from application.jobs.service import JobService
from config import JobsSection
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.document import Document
from domain.knowledge.enums import GENERAL_DOMAIN_KEY, DocumentVisibility
from domain.knowledge.index_sync import VectorOutboxEntry
from domain.runtime.jobs import (
    BackgroundJob,
    BackgroundJobStatus,
    ConcurrencyLimit,
    JobAdmissionError,
    JobConcurrencyLimitReachedError,
    JobType,
)
from domain.shared.exceptions import BusinessRuleViolationError
from infrastructure.database.models import BackgroundJobModel, BenchmarkSweepModel, DocumentModel, UserModel
from infrastructure.metrics import metrics
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


@pytest.mark.parametrize("scope", ["global", "principal"])
async def test_concurrent_submissions_cannot_exceed_backlog(
    pg_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    settings = JobsSection(job_backlog_limit=1 if scope == "global" else 100, job_principal_backlog_limit=1)
    monkeypatch.setattr(
        "infrastructure.repositories.misc.sqlalchemy_background_job_repository.process_section",
        lambda _: settings,
    )
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    barrier = asyncio.Barrier(2)

    async def submit(user: int) -> str:
        try:
            async with sessions.begin() as session:
                await barrier.wait()
                await SQLAlchemyBackgroundJobRepository(session).create(
                    BackgroundJob(
                        job_type=JobType.INGEST_FILE,
                        dispatch_payload={"cost_principal": f"user:{user}"},
                    )
                )
        except JobAdmissionError as exc:
            return exc.scope
        else:
            return "accepted"

    accepted = await asyncio.wait_for(asyncio.gather(submit(1), submit(2 if scope == "global" else 1)), 10)
    assert set(accepted) == {"accepted", scope}
    async with sessions() as session:
        assert await SQLAlchemyBackgroundJobRepository(session).count_active() == 1


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


async def test_concurrent_principal_claims_leave_capacity_for_another_user(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBackgroundJobRepository(session)
        jobs = [
            await repo.create(
                BackgroundJob(job_type=JobType.BENCHMARK, dispatch_payload={"cost_principal": f"user:{user}"})
            )
            for user in (1, 1, 2)
        ]
    barrier = asyncio.Barrier(3)
    limit = ConcurrencyLimit(frozenset({JobType.BENCHMARK}), max_running=3, max_per_principal=1)

    async def claim(job: BackgroundJob) -> bool:
        assert job.id is not None
        try:
            async with sessions.begin() as session:
                await barrier.wait()
                await SQLAlchemyBackgroundJobRepository(session).claim(
                    job.id, f"owner:{job.id}", lease_seconds=120, limit=limit
                )
        except JobConcurrencyLimitReachedError:
            return False
        else:
            return True

    accepted = await asyncio.wait_for(asyncio.gather(*(claim(job) for job in jobs)), 10)
    assert sum(accepted[:2]) == 1
    assert accepted[2] is True


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


async def test_two_workers_have_only_one_execution_owner(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        job = await SQLAlchemyBackgroundJobRepository(session).create(
            BackgroundJob(job_type=JobType.BENCHMARK)
        )
    assert job.id is not None
    job_id = job.id
    barrier = asyncio.Barrier(2)

    async def claim(owner: str) -> int | None:
        async with sessions.begin() as session:
            await barrier.wait()
            return await SQLAlchemyBackgroundJobRepository(session).claim(job_id, owner, lease_seconds=120)

    results = await asyncio.wait_for(asyncio.gather(claim("one"), claim("two")), timeout=5)
    assert results.count(1) == 1
    assert results.count(None) == 1


async def test_old_execution_cannot_commit_business_changes(pg_engine: AsyncEngine) -> None:  # noqa: PLR0915 — one end-to-end fencing scenario
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    factory = UnitOfWorkFactory(SessionDatabase(sessions))
    async with sessions.begin() as session:
        job = await SQLAlchemyBackgroundJobRepository(session).create(
            BackgroundJob(job_type=JobType.BENCHMARK)
        )
        document = await SQLAlchemyDocumentRepository(session).save(Document(filename="original.txt"))
    assert job.id is not None
    assert document.id is not None
    async with sessions.begin() as session:
        repo = SQLAlchemyBackgroundJobRepository(session)
        await repo.claim(job.id, "old", lease_seconds=120)
    async with sessions.begin() as session:
        row = await session.get(BackgroundJobModel, job.id)
        assert row is not None
        row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    async with sessions.begin() as session:
        repo = SQLAlchemyBackgroundJobRepository(session)
        assert [expired.id for expired in await repo.lock_expired(timeout_minutes=30)] == [job.id]
        await repo.release_expired(job.id, error="lease expired", retry_seconds=0)
    async with sessions.begin() as session:
        assert await SQLAlchemyBackgroundJobRepository(session).claim(job.id, "new", lease_seconds=120) == 2
    token = current_execution.set(JobExecution(job.id, "old"))
    try:
        with pytest.raises(ExecutionLeaseLostError):
            async with factory.create(master=True) as uow:
                await uow.documents.update_status(document.id, status="failed", error="stale worker")
    finally:
        current_execution.reset(token)
    async with sessions() as session:
        saved = await SQLAlchemyDocumentRepository(session).get_by_id(document.id)
        assert saved is not None
        assert saved.status.value == "pending"


async def test_execution_guards_share_reads_and_block_attempt_updates(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBackgroundJobRepository(session)
        job = await repo.create(BackgroundJob(job_type=JobType.BENCHMARK))
        assert job.id is not None
        assert await repo.claim(job.id, "owner", lease_seconds=120) == 1

    async with sessions.begin() as first, sessions.begin() as second:
        await SQLAlchemyBackgroundJobRepository(first).lock_execution(job.id, "owner")
        # Parallel/nested business UoWs of the same attempt must not deadlock.
        await asyncio.wait_for(
            SQLAlchemyBackgroundJobRepository(second).lock_execution(job.id, "owner"), timeout=2
        )
        async with sessions() as writer:
            await writer.execute(text("SET LOCAL lock_timeout = '100ms'"))
            with pytest.raises(DBAPIError, match="lock timeout"):
                await SQLAlchemyBackgroundJobRepository(writer).finish_attempt(job.id, "owner")

    async with sessions.begin() as session:
        assert await SQLAlchemyBackgroundJobRepository(session).finish_attempt(job.id, "owner")


async def test_sweep_and_durable_submission_roll_back_together(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    factory = UnitOfWorkFactory(SessionDatabase(sessions))
    monkeypatch.setattr(SQLAlchemyBackgroundJobRepository, "create", AsyncMock(side_effect=ConnectionError()))
    with pytest.raises(ConnectionError):
        await BenchmarkSweepService(factory).create(SweepCreateDTO(), submit_job=True)
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(BenchmarkSweepModel)) == 0
        assert await session.scalar(select(func.count()).select_from(BackgroundJobModel)) == 0


async def test_queue_metrics_are_read_from_committed_database_state(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    now = datetime.now(UTC)
    async with sessions.begin() as session:
        session.add(
            BackgroundJobModel(
                job_type=JobType.BENCHMARK,
                status="pending",
                creation_date=now - timedelta(minutes=5),
                attempt=2,
                dispatch_attempts=4,
            )
        )
        session.add(
            BackgroundJobModel(
                job_type=JobType.SWEEP,
                status="running",
                creation_date=now - timedelta(minutes=10),
                started_at=now - timedelta(minutes=8),
                lease_until=now - timedelta(minutes=1),
            )
        )
    monkeypatch.setattr(metrics, "database", SessionDatabase(sessions))
    await metrics.collect_job_metrics()
    assert metrics.JOB_OLDEST_PENDING.labels(JobType.BENCHMARK)._value.get() >= 300
    assert metrics.JOB_WAIT.labels(JobType.SWEEP)._value.get() == 120
    assert metrics.JOB_EXPIRED._value.get() == 1
    assert metrics.JOB_RETRIES._value.get() == 1
    assert metrics.JOB_DELIVERIES._value.get() == 4


async def test_concurrent_ingest_submission_is_deduplicated_in_database(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    service = JobService(UnitOfWorkFactory(SessionDatabase(sessions)))
    payload: dict[str, object] = {"resolved_dir": "docs/", "reset": False, "domain": "general"}
    results = await asyncio.gather(
        service.create_job(JobType.INGEST, dispatch_payload=payload),
        service.create_job(JobType.INGEST, dispatch_payload=payload),
        return_exceptions=True,
    )
    assert sum(isinstance(result, int) for result in results) == 1
    assert sum(isinstance(result, BusinessRuleViolationError) for result in results) == 1
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(BackgroundJobModel)) == 1


async def test_concurrent_claims_respect_cluster_wide_limit(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repo = SQLAlchemyBackgroundJobRepository(session)
        jobs = [
            await repo.create(BackgroundJob(job_type=job_type))
            for job_type in (JobType.INGEST, JobType.INGEST_FILE, JobType.INGEST)
        ]
    limit = ConcurrencyLimit(frozenset({JobType.INGEST, JobType.INGEST_FILE}), max_running=1)
    barrier = asyncio.Barrier(len(jobs))

    async def claim(job_id: int) -> int | str | None:
        async with sessions.begin() as session:
            await barrier.wait()
            try:
                return await SQLAlchemyBackgroundJobRepository(session).claim(
                    job_id, f"owner-{job_id}", lease_seconds=120, limit=limit
                )
            except JobConcurrencyLimitReachedError:
                return "busy"

    results = await asyncio.wait_for(
        asyncio.gather(*(claim(job.id) for job in jobs if job.id is not None)), timeout=10
    )
    assert results.count(1) == 1, "exactly one replica may start a job of the group"
    assert results.count("busy") == len(jobs) - 1
