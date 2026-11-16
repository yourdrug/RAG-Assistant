"""Admission, fairness and operator actions against the real job repository."""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from config import JobsSection
from domain.runtime.jobs import (
    BackgroundJob,
    ConcurrencyLimit,
    JobAdmissionError,
    JobConcurrencyLimitReachedError,
)
from infrastructure.database.models import BackgroundJobModel
from infrastructure.repositories.misc import sqlalchemy_background_job_repository as jobs_module


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
def repository(monkeypatch: pytest.MonkeyPatch) -> Any:
    engine = create_engine("sqlite://")
    BackgroundJobModel.metadata.tables["background_jobs"].create(engine)
    settings = JobsSection(job_backlog_limit=3, job_principal_backlog_limit=2)
    monkeypatch.setattr(jobs_module, "process_section", lambda _: settings, raising=False)
    with Session(engine) as session:

        class Facade:
            def add(self, value: object) -> None:
                session.add(value)

            def get_bind(self) -> Any:
                return session.get_bind()

            async def execute(self, statement: Any) -> Any:
                return session.execute(statement)

            async def scalar(self, statement: Any) -> Any:
                return session.scalar(statement)

            async def flush(self) -> None:
                session.flush()

            async def refresh(self, value: object) -> None:
                session.refresh(value)

        yield jobs_module.SQLAlchemyBackgroundJobRepository(cast("AsyncSession", Facade())), session
    engine.dispose()


def pending(principal: str) -> BackgroundJob:
    return BackgroundJob(job_type="ingest_file", dispatch_payload={"cost_principal": principal})


@pytest.mark.asyncio
async def test_admission_rejects_principal_and_global_backlog(repository: Any) -> None:
    repo, _ = repository
    await repo.create(pending("user:1"))
    await repo.create(pending("user:1"))
    with pytest.raises(JobAdmissionError) as personal:
        await repo.create(pending("user:1"))
    assert personal.value.scope == "principal"
    await repo.create(pending("user:2"))
    with pytest.raises(JobAdmissionError) as global_limit:
        await repo.create(pending("user:3"))
    assert global_limit.value.scope == "global"


@pytest.mark.asyncio
async def test_claim_limits_one_principal_without_blocking_another(repository: Any) -> None:
    repo, _ = repository
    first = await repo.create(pending("user:1"))
    second = await repo.create(pending("user:1"))
    other = await repo.create(pending("user:2"))
    limit = ConcurrencyLimit(frozenset({"ingest_file"}), max_running=3, max_per_principal=1)
    assert await repo.claim(first.id, "a", lease_seconds=180, limit=limit) == 1
    with pytest.raises(JobConcurrencyLimitReachedError):
        await repo.claim(second.id, "b", lease_seconds=180, limit=limit)
    assert await repo.claim(other.id, "c", lease_seconds=180, limit=limit) == 1
    assert (await repo.get_by_id(second.id)).attempt == 0


@pytest.mark.asyncio
async def test_cancel_fences_running_attempt_and_prevents_republication(repository: Any) -> None:
    repo, _ = repository
    job = await repo.create(pending("user:1"))
    await repo.claim(job.id, "owner", lease_seconds=180)
    assert await repo.cancel(job.id)
    assert not await repo.finish_attempt(job.id, "owner")
    assert not await repo.renew_lease(job.id, "owner", lease_seconds=180)
    assert await repo.claim(job.id, "duplicate", lease_seconds=180) is None
    assert await repo.pending_dispatch() == []
    assert (await repo.get_by_id(job.id)).status == "cancelled"


@pytest.mark.asyncio
async def test_relay_rotates_between_principals_before_the_batch_limit(repository: Any) -> None:
    repo, session = repository

    for job_id in range(1, 101):
        session.add(
            BackgroundJobModel(
                id=job_id,
                job_type="ingest_file",
                status="pending",
                dispatch_payload={"cost_principal": "user:1"},
            )
        )

    session.add(
        BackgroundJobModel(
            id=101, job_type="ingest_file", status="pending", dispatch_payload={"cost_principal": "user:2"}
        )
    )
    session.flush()
    selected = await repo.pending_dispatch(limit=2)
    assert {job.id for job in selected} == {1, 101}


@pytest.mark.asyncio
async def test_cleanup_is_bounded_and_preserves_active_work(repository: Any) -> None:
    repo, session = repository
    old = datetime.now(UTC) - timedelta(days=50)

    for job_id, state in enumerate(["done", "failed", "cancelled", "running"], 1):
        session.add(BackgroundJobModel(id=job_id, job_type="ingest_file", status=state, creation_date=old))

    session.flush()
    assert await repo.delete_old(days=30, limit=2) == 2
    assert await repo.delete_old(days=30, limit=2) == 1
    assert (await repo.get_by_id(4)).status == "running"
