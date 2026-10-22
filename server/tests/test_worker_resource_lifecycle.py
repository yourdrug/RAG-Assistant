"""Worker producers and evaluation tasks share their process-owned resources."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from application.benchmark import BenchmarkService
from infrastructure.benchmark import runner
from infrastructure.worker import queue, runtime, tasks


@pytest.fixture
def pool_state(monkeypatch):
    monkeypatch.setattr(queue, "_pool", None)
    if hasattr(queue, "pool_lock"):
        monkeypatch.setattr(queue, "pool_lock", asyncio.Lock())


@pytest.mark.asyncio
async def test_concurrent_first_enqueues_create_only_one_pool(monkeypatch, pool_state):
    pools = []

    async def create(settings):
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
    assert queue._pool is None


@pytest.mark.asyncio
async def test_shutdown_waits_for_inflight_pool_creation(monkeypatch, pool_state):
    started = asyncio.Event()
    release = asyncio.Event()
    pool = SimpleNamespace(enqueue_job=AsyncMock(), close=AsyncMock())

    async def create(settings):
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
    assert queue._pool is None


@pytest.mark.asyncio
async def test_concurrent_shutdown_closes_shared_pool_once(monkeypatch, pool_state):
    async def close():
        await asyncio.sleep(0)

    pool = SimpleNamespace(close=AsyncMock(side_effect=close))
    monkeypatch.setattr(queue, "_pool", pool)
    await asyncio.gather(queue.close_arq_pool(), queue.close_arq_pool())
    pool.close.assert_awaited_once()
    assert queue._pool is None


@pytest.mark.asyncio
async def test_worker_benchmark_runs_container_rag_pipeline(monkeypatch):
    rag = object()
    factory = object()
    monkeypatch.setattr(
        tasks,
        "BenchmarkResultService",
        lambda factory: SimpleNamespace(get_job_result=AsyncMock(return_value=None)),
    )
    container = SimpleNamespace(
        infrastructure=SimpleNamespace(db=SimpleNamespace(uow_factory=factory)),
        application=SimpleNamespace(rag_service=rag),
    )
    questions = [{"id": 42, "question": "Which provision applies?"}]
    monkeypatch.setattr(tasks, "load_benchmark_questions", AsyncMock(return_value=questions))
    used = []

    async def evaluate(**kwargs):
        used.append(kwargs.get("rag_service"))
        raise RuntimeError("stop after selecting pipeline")

    async def tracked(factory_arg, job_id, action, **kwargs):
        with pytest.raises(RuntimeError, match="stop after selecting pipeline"):
            await action()

    monkeypatch.setattr(runtime, "run_tracked_job", tracked)
    container.infrastructure.ml = SimpleNamespace(
        benchmark_service=BenchmarkService(rag_service=rag, runner=SimpleNamespace(run=evaluate))
    )
    await tasks.run_benchmark(
        {"container": container}, dataset="custom", out_dir="unused", top_k=4, judge_model="judge", job_id=8
    )
    assert used == [rag]


@pytest.mark.asyncio
async def test_benchmark_requires_rag_pipeline(monkeypatch):
    with pytest.raises(ValueError, match="rag_service is required"):
        await runner.run_benchmark_async([{"question": "q"}], 4, "judge")
