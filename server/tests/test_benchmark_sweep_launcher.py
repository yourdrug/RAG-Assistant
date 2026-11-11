"""A sweep that cannot be queued must not stay pending and block every later sweep."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from application.benchmark import BenchmarkSweepLauncher, BenchmarkSweepService
from application.benchmark.dto import SweepCreateDTO
from domain.benchmark.sweep import BenchmarkSweepStatus
from domain.runtime.jobs import BackgroundJobStatus, JobType
from tests.fakes import FakeUnitOfWorkFactory


def launcher_with(queue: object) -> tuple[BenchmarkSweepLauncher, FakeUnitOfWorkFactory]:
    factory = FakeUnitOfWorkFactory()
    return BenchmarkSweepLauncher(factory, BenchmarkSweepService(factory), queue), factory  # type: ignore[arg-type]


def body() -> SweepCreateDTO:
    return SweepCreateDTO(search_space={"top_k": {"values": [2]}})


@pytest.mark.asyncio
async def test_started_sweep_is_linked_to_a_queued_job() -> None:
    queue = SimpleNamespace(enqueue_sweep=AsyncMock())
    launcher, factory = launcher_with(queue)

    sweep, job_id = await launcher.start(body())

    assert sweep.id is not None
    job = factory._uow.fake_background_jobs.jobs[job_id]
    assert (job.job_type, job.status) == (JobType.SWEEP, BackgroundJobStatus.PENDING)
    assert job.related_id == sweep.id
    assert factory._uow.fake_benchmark_sweeps.job_ids[sweep.id] == job_id
    queue.enqueue_sweep.assert_awaited_once_with(sweep_id=sweep.id, job_id=job_id)


@pytest.mark.asyncio
async def test_enqueue_failure_fails_job_and_sweep() -> None:
    queue = SimpleNamespace(enqueue_sweep=AsyncMock(side_effect=ConnectionError("redis down")))
    launcher, factory = launcher_with(queue)
    jobs, sweeps = factory._uow.fake_background_jobs, factory._uow.fake_benchmark_sweeps
    jobs.mark_failed = AsyncMock()  # type: ignore[method-assign]
    sweeps.update_status = AsyncMock()  # type: ignore[method-assign]

    with pytest.raises(ConnectionError):
        await launcher.start(body())

    jobs.mark_failed.assert_awaited_once_with(1, "Failed to enqueue benchmark sweep")
    sweeps.update_status.assert_awaited_once_with(1, BenchmarkSweepStatus.FAILED.value)


@pytest.mark.asyncio
async def test_job_creation_failure_still_fails_sweep() -> None:
    queue = SimpleNamespace(enqueue_sweep=AsyncMock())
    launcher, factory = launcher_with(queue)
    jobs, sweeps = factory._uow.fake_background_jobs, factory._uow.fake_benchmark_sweeps
    jobs.create = AsyncMock(side_effect=RuntimeError("db"))  # type: ignore[method-assign]
    jobs.mark_failed = AsyncMock()  # type: ignore[method-assign]
    sweeps.update_status = AsyncMock()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="db"):
        await launcher.start(body())

    jobs.mark_failed.assert_not_awaited()
    sweeps.update_status.assert_awaited_once_with(1, BenchmarkSweepStatus.FAILED.value)
    queue.enqueue_sweep.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_compensation_keeps_the_enqueue_error() -> None:
    queue = SimpleNamespace(enqueue_sweep=AsyncMock(side_effect=ConnectionError("redis down")))
    launcher, factory = launcher_with(queue)
    factory._uow.fake_benchmark_sweeps.update_status = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("db down")
    )

    with pytest.raises(ConnectionError, match="redis down"):
        await launcher.start(body())
