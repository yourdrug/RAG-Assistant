"""Worker producers and evaluation tasks share their process-owned resources."""

from __future__ import annotations

import asyncio
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock

import pytest

from application.benchmark import BenchmarkService, jobs as benchmark_jobs
from application.benchmark.jobs import BenchmarkJobService
from infrastructure.benchmark.corpus import benchmark_scope
from infrastructure.benchmark.evaluation import runner
from infrastructure.worker import queue, runtime, tasks


if TYPE_CHECKING:
    from application.benchmark.export_service import BenchmarkExportService
    from application.benchmark.result_service import BenchmarkResultService
    from application.chat.ports import ChatRAGPort
    from application.uow.factory import UnitOfWorkFactory


@pytest.fixture
def pool_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(queue.pool_state, "pool", None)

    if hasattr(queue, "pool_lock"):
        monkeypatch.setattr(queue, "pool_lock", asyncio.Lock())


@pytest.mark.asyncio
async def test_concurrent_first_enqueues_create_only_one_pool(
    monkeypatch: pytest.MonkeyPatch, pool_state: object
) -> None:
    pools = []

    async def create(settings: Any) -> Any:
        pool = SimpleNamespace(enqueue_job=AsyncMock(), close=AsyncMock())
        pools.append(pool)
        await asyncio.sleep(0)
        return pool

    monkeypatch.setattr(queue, "create_pool", create)
    await asyncio.gather(*(queue.enqueue_sweep(sweep_id=i, job_id=i) for i in range(12)))
    await queue.close_arq_pool()

    assert len(pools) == 1
    assert pools[0].enqueue_job.await_count == 12
    pools[0].close.assert_awaited_once()
    assert queue.pool_state.pool is None


@pytest.mark.asyncio
async def test_shutdown_waits_for_inflight_pool_creation(
    monkeypatch: pytest.MonkeyPatch, pool_state: object
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    pool = SimpleNamespace(enqueue_job=AsyncMock(), close=AsyncMock())

    async def create(settings: object) -> Any:
        started.set()
        await release.wait()
        return pool

    monkeypatch.setattr(queue, "create_pool", create)
    enqueue = asyncio.create_task(queue.enqueue_sweep(sweep_id=1, job_id=2))
    await started.wait()
    shutdown = asyncio.create_task(queue.close_arq_pool())
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(enqueue, shutdown)

    pool.close.assert_awaited_once()
    assert queue.pool_state.pool is None


@pytest.mark.asyncio
async def test_concurrent_shutdown_closes_shared_pool_once(
    monkeypatch: pytest.MonkeyPatch, pool_state: object
) -> None:
    async def close() -> None:
        await asyncio.sleep(0)

    pool = SimpleNamespace(close=AsyncMock(side_effect=close))
    monkeypatch.setattr(queue.pool_state, "pool", pool)
    await asyncio.gather(queue.close_arq_pool(), queue.close_arq_pool())
    pool.close.assert_awaited_once()
    assert queue.pool_state.pool is None


@pytest.mark.asyncio
async def test_worker_benchmark_runs_container_rag_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    rag = object()
    factory = object()
    results = SimpleNamespace(get_job_result=AsyncMock(return_value=None))
    container = SimpleNamespace(
        infrastructure=SimpleNamespace(db=SimpleNamespace(uow_factory=factory)),
        application=SimpleNamespace(rag_service=rag),
    )
    questions = [{"id": 42, "question": "Which provision applies?"}]
    monkeypatch.setattr(benchmark_jobs, "load_benchmark_questions", AsyncMock(return_value=questions))
    used = []

    async def evaluate(**kwargs: Any) -> None:
        used.append(kwargs.get("rag_service"))
        msg = "stop after selecting pipeline"
        raise RuntimeError(msg)

    async def tracked(factory_arg: object, job_id: object, action: Any, **kwargs: Any) -> None:
        with pytest.raises(RuntimeError, match="stop after selecting pipeline"):
            await action()

    monkeypatch.setattr(runtime, "run_tracked_job", tracked)
    container.infrastructure.ml = SimpleNamespace(
        ml_clients=None,
        benchmark_service=BenchmarkService(
            rag_service=cast("ChatRAGPort", rag), runner=SimpleNamespace(run=evaluate)
        ),
    )
    await tasks.run_benchmark(
        {
            "uow_factory": factory,
            "benchmark_job": BenchmarkJobService(
                cast("UnitOfWorkFactory", factory),
                container.infrastructure.ml.benchmark_service,
                cast("BenchmarkResultService", results),
                cast("BenchmarkExportService", object()),
                partial(benchmark_scope, None),
            ),
        },
        dataset="custom",
        out_dir="unused",
        top_k=4,
        judge_model="judge",
        job_id=8,
    )
    assert used == [rag]


@pytest.mark.asyncio
async def test_benchmark_requires_rag_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="rag_service is required"):
        await runner.run_benchmark_async([{"question": "q"}], 4, "judge")
