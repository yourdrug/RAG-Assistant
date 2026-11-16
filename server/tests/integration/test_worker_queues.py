"""A saturated heavy queue cannot occupy maintenance execution slots."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

from arq import create_pool
from arq.worker import Worker, func
import pytest
from tests.fakes import FakeUnitOfWorkFactory

from application.jobs.service import JobService
from domain.runtime.jobs import BackgroundJob, JobType
from infrastructure.adapters import job_enqueuer_adapter
from infrastructure.worker.queue import arq_redis_settings
from infrastructure.worker.runtime import job


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_owner_cancellation_aborts_real_arq_work_without_republishing(  # noqa: PLR0915 -- real worker lifecycle
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = os.environ.get("TEST_WORKER_REDIS_URL")

    if not url:
        pytest.skip("Set TEST_WORKER_REDIS_URL for isolated Redis worker tests")

    pool = await create_pool(arq_redis_settings(url))
    factory = FakeUnitOfWorkFactory()
    repo = factory._uow.fake_background_jobs
    durable = await repo.create(
        BackgroundJob(job_type=JobType.INGEST_FILE, dispatch_payload={"cost_principal": "user:1"})
    )
    assert durable.id is not None
    started, cancelled = asyncio.Event(), asyncio.Event()

    @job(JobType.INGEST_FILE)
    async def work(ctx: dict[str, Any], *, job_id: int) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    worker = Worker(
        [func(work, name="cancel_test", keep_result=0)],
        redis_pool=pool,
        queue_name="batch_tasks",
        burst=True,
        poll_delay=0.01,
        handle_signals=False,
        allow_abort_jobs=True,
        keep_result=0,
        ctx={"uow_factory": factory},
    )
    monkeypatch.setattr(job_enqueuer_adapter, "get_pool", AsyncMock(return_value=pool))
    task = None
    try:
        await pool.enqueue_job(
            "cancel_test", job_id=durable.id, _job_id=f"job:{durable.id}", _queue_name="batch_tasks"
        )
        task = asyncio.create_task(worker.async_run())
        await asyncio.wait_for(started.wait(), 5)
        await JobService(factory, aborter=job_enqueuer_adapter.JobEnqueuerAdapter()).cancel(
            durable.id, user_id=1
        )
        await asyncio.wait_for(cancelled.wait(), 5)
        await asyncio.wait_for(task, 5)
        assert repo.jobs[durable.id].status == "cancelled"
        assert await repo.pending_dispatch() == []
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        await worker.close()


async def test_maintenance_runs_with_all_heavy_slots_busy() -> None:  # noqa: PLR0915 — exercise both real workers
    url = os.environ.get("TEST_WORKER_REDIS_URL")

    if not url:
        pytest.skip("Set TEST_WORKER_REDIS_URL for isolated Redis worker tests")

    settings = arq_redis_settings(url)
    namespace = uuid4().hex
    heavy_queue, maintenance_queue = f"{namespace}:heavy", f"{namespace}:maintenance"
    busy = asyncio.Event()
    release = asyncio.Event()
    running = 0

    async def heavy(ctx: dict[str, Any]) -> None:
        nonlocal running
        running += 1

        if running == 4:
            busy.set()

        await release.wait()

    async def maintenance(ctx: dict[str, Any]) -> None:
        assert running == 4
        assert not release.is_set()
        release.set()

    pool = await create_pool(settings)
    workers = [
        Worker(
            [func(heavy, name="heavy")],
            redis_settings=settings,
            queue_name=heavy_queue,
            max_jobs=4,
            burst=True,
            poll_delay=0.01,
            handle_signals=False,
            keep_result=0,
        ),
        Worker(
            [func(maintenance, name="maintenance")],
            redis_settings=settings,
            queue_name=maintenance_queue,
            max_jobs=1,
            burst=True,
            poll_delay=0.01,
            handle_signals=False,
            keep_result=0,
        ),
    ]
    try:
        for _ in range(4):
            await pool.enqueue_job("heavy", _queue_name=heavy_queue)

        task = asyncio.create_task(workers[0].async_run())
        await asyncio.wait_for(busy.wait(), timeout=5)
        await pool.enqueue_job("maintenance", _queue_name=maintenance_queue)
        await asyncio.wait_for(workers[1].async_run(), timeout=5)
        await asyncio.wait_for(task, timeout=5)
        assert workers[0].jobs_complete == 4
        assert workers[1].jobs_complete == 1
    finally:
        release.set()
        await asyncio.gather(*(worker.close() for worker in workers))
        await pool.aclose()
