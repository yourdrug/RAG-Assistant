"""Job owners and administrators operate on durable state before broker delivery."""

from unittest.mock import AsyncMock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from application.jobs.service import JobService
from domain.runtime.jobs import BackgroundJob, BackgroundJobStatus
from domain.shared.exceptions import EntityNotFoundError
from infrastructure.adapters import job_enqueuer_adapter
from tests.fakes import FakeUnitOfWorkFactory


@pytest.mark.asyncio
async def test_abort_translates_broker_failure_to_the_application_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(job_enqueuer_adapter, "get_pool", AsyncMock(side_effect=RedisConnectionError()))
    with pytest.raises(ConnectionError):
        await job_enqueuer_adapter.JobEnqueuerAdapter().abort_job(1, "ingest", None)


@pytest.mark.asyncio
async def test_cancel_checks_owner_and_survives_broker_unavailability() -> None:
    factory = FakeUnitOfWorkFactory()
    job = await factory._uow.background_jobs.create(
        BackgroundJob(
            job_type="ingest",
            dispatch_payload={"cost_principal": "user:1"},
        )
    )
    assert job.id is not None
    aborter = AsyncMock()
    aborter.abort_job.side_effect = ConnectionError("broker unavailable")
    service = JobService(factory, aborter=aborter)
    with pytest.raises(EntityNotFoundError):
        await service.cancel(job.id, user_id=2)
    with pytest.raises(EntityNotFoundError):
        await service.cancel(job.id + 1000, user_id=2)
    assert job.status == "pending"
    await service.cancel(job.id, user_id=1)
    assert factory._uow.fake_background_jobs.jobs[job.id].status == "cancelled"


@pytest.mark.asyncio
async def test_retry_creates_a_new_job_and_keeps_ingestion_progress() -> None:
    factory = FakeUnitOfWorkFactory()
    old = await factory._uow.background_jobs.create(
        BackgroundJob(
            job_type="ingest",
            status=BackgroundJobStatus.FAILED,
            dispatch_payload={"cost_principal": "user:1", "reset": True},
            progress={"docs/a.txt": "hash-a"},
        )
    )
    assert old.id is not None
    new_id = await JobService(factory).retry(old.id)
    assert new_id != old.id
    new = factory._uow.fake_background_jobs.jobs[new_id]
    assert new.status == "pending"
    assert new.progress == {"docs/a.txt": "hash-a"}
    assert new.dispatch_payload == old.dispatch_payload
    assert new.deduplication_key == f"retry:{old.id}"
    assert factory._uow.fake_background_jobs.jobs[old.id].status == "failed"
