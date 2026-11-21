"""Failure boundaries for durable jobs, execution leases and isolated maintenance."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock

import httpx
from openai import APIStatusError
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from application.documents.processor import DocumentProcessor
from composition.cli.commands import worker as command
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus
from domain.runtime.jobs import BackgroundJob, JobType
from domain.shared.exceptions import PermissionDeniedError
from infrastructure.database.models import BackgroundJobModel
from infrastructure.repositories.misc.sqlalchemy_background_job_repository import (
    SQLAlchemyBackgroundJobRepository,
)
from infrastructure.scheduling.scheduler import Scheduler
from infrastructure.worker import cron, document_dispatch, policy, runtime
from infrastructure.worker.policy import RetryPolicy
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from sqlalchemy.ext.asyncio import AsyncSession

    from application.uow import UnitOfWorkFactory

JobHarness = tuple[SQLAlchemyBackgroundJobRepository, Session]


@pytest.fixture
def jobs() -> Iterator[JobHarness]:
    engine = create_engine("sqlite://")
    BackgroundJobModel.metadata.tables[BackgroundJobModel.__tablename__].create(engine)
    with Session(engine) as session:

        class AsyncSessionFacade:
            async def execute(self, statement: Any) -> Any:
                return session.execute(statement)

            async def flush(self) -> None:
                session.flush()

        session.add(BackgroundJobModel(id=1, job_type=JobType.DOCUMENT_DELETION, status="pending"))
        session.commit()
        repo = SQLAlchemyBackgroundJobRepository(cast("AsyncSession", AsyncSessionFacade()))
        yield repo, session
    engine.dispose()


@pytest.mark.asyncio
async def test_claim_is_atomic_and_terminal_redelivery_does_nothing(jobs: JobHarness) -> None:
    repo, _ = jobs
    assert await repo.claim(1, "first", lease_seconds=120) == 1
    assert await repo.claim(1, "duplicate", lease_seconds=120) is None
    assert await repo.finish_attempt(1, "first")
    assert await repo.claim(1, "redelivery", lease_seconds=120) is None


@pytest.mark.asyncio
async def test_expired_owner_cannot_heartbeat_or_finish_new_attempt(jobs: JobHarness) -> None:
    repo, session = jobs
    assert await repo.claim(1, "old", lease_seconds=120) == 1
    row = session.get(BackgroundJobModel, 1)
    assert row is not None
    row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    session.flush()
    assert await repo.claim(1, "new", lease_seconds=120) is None, "only recovery releases a lost attempt"
    assert [job.id for job in await repo.lock_expired(timeout_minutes=30)] == [1]
    await repo.release_expired(1, error="lease expired", retry_seconds=0)
    assert await repo.claim(1, "new", lease_seconds=120) == 2
    assert not await repo.renew_lease(1, "old", lease_seconds=120)
    assert not await repo.finish_attempt(1, "old", error="stale failure")
    assert await repo.finish_attempt(1, "new")
    session.expire_all()
    assert session.scalar(select(BackgroundJobModel.status)) == "done"


@pytest.mark.asyncio
async def test_expired_lease_cannot_write_even_before_reclaim(jobs: JobHarness) -> None:
    repo, session = jobs
    await repo.claim(1, "old", lease_seconds=120)
    row = session.get(BackgroundJobModel, 1)
    assert row is not None
    row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    session.flush()
    assert not await repo.finish_attempt(1, "old")
    assert not await repo.renew_lease(1, "old", lease_seconds=120)


@pytest.mark.asyncio
async def test_unknown_or_invalid_durable_payload_is_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = FakeUnitOfWorkFactory()
    repo = factory._uow.fake_background_jobs
    await repo.create(BackgroundJob(job_type="missing", dispatch_payload={}))
    await repo.create(
        BackgroundJob(job_type=JobType.DOCUMENT_DELETION, dispatch_payload={"document_id": "bad"})
    )
    publish = AsyncMock()
    monkeypatch.setattr(document_dispatch, "enqueue_arq", publish)
    await document_dispatch.cron_dispatch_documents({"uow_factory": factory})
    assert all(job.status == "failed" for job in repo.jobs.values())
    publish.assert_not_awaited()


def test_worker_roles_have_disjoint_queues_and_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = Mock(return_value=Mock())
    monkeypatch.setattr(command, "Worker", worker)
    roles = {}

    for role in command.WorkerRole:
        command.worker(role=role)
        roles[role] = worker.call_args.kwargs

    documents, batch = roles[command.WorkerRole.DOCUMENTS], roles[command.WorkerRole.BATCH]
    maintenance = roles[command.WorkerRole.MAINTENANCE]
    assert len({kwargs["queue_name"] for kwargs in roles.values()}) == 3
    assert documents["cron_jobs"] == batch["cron_jobs"] == []
    assert maintenance["cron_jobs"]
    assert maintenance["functions"] == []
    document_names = {f.name for f in documents["functions"]}
    batch_names = {f.name for f in batch["functions"]}
    assert document_names == {"process_document", "delete_document"}
    assert batch_names == {"run_full_ingest", "run_single_ingest", "run_benchmark", "run_sweep", "run_sync"}
    assert all(kwargs["keep_result"] == 0 for kwargs in roles.values())


def test_api_scheduler_only_schedules_process_local_work() -> None:
    scheduler = Scheduler()
    scheduler.configure()
    assert {job.id for job in cast("Any", scheduler.scheduler).get_jobs()} == {
        "config_resync",
        "infra_metrics_collector",
    }


@pytest.mark.asyncio
async def test_permission_failure_is_not_retried(jobs: JobHarness) -> None:
    repo, session = jobs

    @asynccontextmanager
    async def create(**kwargs: object) -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(background_jobs=repo)
        session.commit()

    factory = SimpleNamespace(create=create)

    async def denied() -> None:
        raise PermissionDeniedError

    await runtime.run_tracked_job(
        cast("UnitOfWorkFactory", factory), 1, denied, description="delete_document"
    )
    assert session.scalar(select(BackgroundJobModel.status)) == "failed"


@pytest.mark.asyncio
async def test_delivery_after_committed_document_effect_skips_processing() -> None:

    factory = FakeUnitOfWorkFactory()
    doc = await factory._uow.documents.save(Document(filename="done.txt", status=DocumentStatus.INDEXING))
    processor = SimpleNamespace(_uow_factory=factory)
    assert doc.id is not None
    assert not await DocumentProcessor.start_processing(
        cast("DocumentProcessor", processor), doc.id, skip_persisted=True
    )
    assert doc.status == DocumentStatus.INDEXING


@pytest.mark.parametrize("status,retry", [(429, True), (503, True), (400, False), (403, False)])
def test_provider_status_controls_execution_retry(status: int, retry: bool) -> None:
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://provider.invalid"), headers={"retry-after": "30"}
    )
    error = APIStatusError("provider error", response=response, body=None)
    delay = RetryPolicy().delay(error, 1)
    assert (delay is not None) is retry

    if delay is not None:
        assert 30 <= delay <= 35


@pytest.mark.asyncio
async def test_unsupported_payload_version_does_not_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = FakeUnitOfWorkFactory()
    job = await factory._uow.fake_background_jobs.create(
        BackgroundJob(
            job_type=JobType.DOCUMENT_DELETION,
            payload_version=99,
            dispatch_payload={"document_id": 1, "user_id": 1},
        )
    )
    publish = AsyncMock()
    monkeypatch.setattr(document_dispatch, "enqueue_arq", publish)
    await document_dispatch.cron_dispatch_documents({"uow_factory": factory})
    assert job.status == "failed"
    publish.assert_not_awaited()


async def seed_expired_processing_job(factory: FakeUnitOfWorkFactory, *, attempt: int, job_type: str) -> int:
    document = await factory._uow.fake_documents.save(
        Document(filename="lost.pdf", status=DocumentStatus.PROCESSING)
    )
    job = await factory._uow.fake_background_jobs.create(
        BackgroundJob(job_type=job_type, related_id=document.id)
    )
    assert job.id is not None
    job.mark_running()
    job.attempt = attempt
    job.attempt_id = "lost-worker"
    job.lease_until = datetime.now(UTC) - timedelta(seconds=1)
    assert document.id is not None
    return document.id


@pytest.mark.asyncio
async def test_expired_lease_is_requeued_while_budget_remains() -> None:
    factory = FakeUnitOfWorkFactory()
    document_id = await seed_expired_processing_job(factory, attempt=1, job_type=JobType.DOCUMENT_PROCESSING)
    await cron.cron_recover_orphaned_jobs({"uow_factory": factory})
    job = factory._uow.fake_background_jobs.jobs[1]
    assert job.status == "pending"
    assert job.attempt_id is None, "the lost attempt must be fenced out"
    document = await factory._uow.fake_documents.get_by_id(document_id)
    assert document is not None
    assert document.status == DocumentStatus.PROCESSING


@pytest.mark.asyncio
async def test_expired_lease_on_last_attempt_fails_job_and_document() -> None:
    factory = FakeUnitOfWorkFactory()
    document_id = await seed_expired_processing_job(factory, attempt=3, job_type=JobType.DOCUMENT_PROCESSING)
    await cron.cron_recover_orphaned_jobs({"uow_factory": factory})
    assert factory._uow.fake_background_jobs.jobs[1].status == "failed"
    document = await factory._uow.fake_documents.get_by_id(document_id)
    assert document is not None
    assert document.status == DocumentStatus.FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize("job_type", [JobType.SWEEP, JobType.BENCHMARK])
async def test_expired_non_replayable_job_is_never_repeated_implicitly(job_type: str) -> None:
    factory = FakeUnitOfWorkFactory()
    await seed_expired_processing_job(factory, attempt=1, job_type=job_type)
    await cron.cron_recover_orphaned_jobs({"uow_factory": factory})
    assert factory._uow.fake_background_jobs.jobs[1].status == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("job_type", [JobType.BENCHMARK])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_non_replayable_job_is_terminal_after_failure(job_type: str, cancelled: bool) -> None:
    factory = FakeUnitOfWorkFactory()
    repo = factory._uow.fake_background_jobs
    job = await repo.create(BackgroundJob(job_type=job_type))
    assert job.id is not None
    executed = 0

    @runtime.job(job_type)
    async def action(ctx: dict[str, object], **kwargs: object) -> None:
        nonlocal executed
        executed += 1

        if cancelled:
            raise asyncio.CancelledError

        message = "worker lost connection after side effect"
        raise ConnectionError(message)

    ctx = {"uow_factory": factory}

    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await action(ctx, job_id=job.id)
    else:
        await action(ctx, job_id=job.id)

    assert repo.jobs[job.id].status == "failed"
    await action(ctx, job_id=job.id)
    assert executed == 1
    assert repo.jobs[job.id].attempt == 1


@pytest.mark.asyncio
async def test_relay_counts_only_publications_accepted_as_new(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = FakeUnitOfWorkFactory()
    repo = factory._uow.fake_background_jobs
    await repo.create(
        BackgroundJob(job_type=JobType.DOCUMENT_DELETION, dispatch_payload={"document_id": 1, "user_id": 2})
    )
    # New publication, then duplicates refused while the job waits in the broker.
    publish = AsyncMock(side_effect=[Mock(), None, None])
    monkeypatch.setattr(document_dispatch, "enqueue_arq", publish)

    for _ in range(3):
        await document_dispatch.cron_dispatch_documents({"uow_factory": factory})

    assert publish.await_count == 3
    assert repo.jobs[1].dispatch_attempts == 1


def test_batch_types_share_cluster_wide_limits() -> None:
    ingest = policy.concurrency_limit(JobType.INGEST_FILE)
    assert ingest is not None
    assert ingest.job_types == {JobType.INGEST, JobType.INGEST_FILE}
    documents = policy.concurrency_limit(JobType.DOCUMENT_PROCESSING)
    assert documents is not None
    assert documents.max_running is None
    assert documents.max_per_principal == 2
    assert policy.max_tries(JobType.SWEEP) == 1
    assert policy.max_tries(JobType.DOCUMENT_PROCESSING) == policy.DEFAULT_MAX_TRIES


@pytest.mark.asyncio
async def test_relay_stops_round_when_broker_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = FakeUnitOfWorkFactory()
    repo = factory._uow.fake_background_jobs

    for document_id in (1, 2, 3):
        await repo.create(
            BackgroundJob(
                job_type=JobType.DOCUMENT_DELETION,
                dispatch_payload={"document_id": document_id, "user_id": 2},
            )
        )

    publish = AsyncMock(side_effect=ConnectionError("Redis unavailable"))
    monkeypatch.setattr(document_dispatch, "enqueue_arq", publish)
    await document_dispatch.cron_dispatch_documents({"uow_factory": factory})
    publish.assert_awaited_once()
    assert all(job.status == "pending" for job in repo.jobs.values()), (
        "committed work stays for the next round"
    )
