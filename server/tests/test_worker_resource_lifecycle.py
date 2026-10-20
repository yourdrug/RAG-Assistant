"""Worker producers and evaluation tasks share their process-owned resources."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from domain.identity.roles import UserKind, UserRole
from domain.rag.context import RagContext
from infrastructure.benchmark import answer_generators, retrieval, runner
from infrastructure.worker import queue, tasks


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

    monkeypatch.setattr(tasks, "run_tracked_job", tracked)
    monkeypatch.setattr(tasks, "run_benchmark_async", evaluate)
    await tasks.run_benchmark(
        {"container": container}, dataset="custom", out_dir="unused", top_k=4, judge_model="judge", job_id=8
    )
    assert used == [rag]


@pytest.mark.asyncio
async def test_benchmark_requires_explicit_standalone_opt_in(monkeypatch):
    with pytest.raises(ValueError, match="rag_service.*standalone"):
        await runner.run_benchmark_async([{"question": "q"}], "unused", 4, "judge", export_files=False)


@pytest.mark.asyncio
async def test_explicit_standalone_closes_owned_resources_on_failure(monkeypatch):
    closed = []

    class Standalone:
        def __init__(self, *args):
            pass

        async def generate(self, question, ctx):
            raise RuntimeError("standalone retrieval failed")

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(answer_generators, "StandaloneBenchmarkGenerator", Standalone)
    with pytest.raises(RuntimeError, match="standalone retrieval failed"):
        await runner.run_benchmark_async(
            [{"question": "q"}], "unused", 4, "judge", export_files=False, standalone=True
        )
    assert closed == [True]


@pytest.mark.asyncio
async def test_standalone_closes_llm_when_retrieval_fails(monkeypatch):
    llm = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(retrieval, "build_llm", lambda *args, **kwargs: llm)

    def fail(*args, **kwargs):
        raise RuntimeError("retrieval failed")

    monkeypatch.setattr(retrieval, "retrieve_with_scores_hybrid", fail)
    ctx = RagContext(user_id=0, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN)
    generator = answer_generators.StandaloneBenchmarkGenerator(4, 12, ctx)
    monkeypatch.setattr(generator._clients, "ensure_bm25_loaded", AsyncMock(return_value=None))
    close = AsyncMock()
    monkeypatch.setattr(generator._clients, "close", close)
    with pytest.raises(RuntimeError, match="retrieval failed"):
        await generator.generate({"question": "q"}, ctx)
    llm.close.assert_awaited_once()
    await generator.close()
    close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_count", [1, 3])
@pytest.mark.parametrize("retrieval_fails", [False, True])
async def test_standalone_cancellation_waits_for_owned_retrieval_before_closing(
    monkeypatch, cancel_count, retrieval_fails
):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    llm = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(retrieval, "build_llm", lambda *args, **kwargs: llm)

    def retrieve(*args, **kwargs):
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("retrieval not released")
        finished.set()
        if retrieval_fails:
            raise RuntimeError("retrieval failed after cancellation")
        return []

    monkeypatch.setattr(retrieval, "retrieve_with_scores_hybrid", retrieve)
    ctx = RagContext(user_id=0, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN)
    generator = answer_generators.StandaloneBenchmarkGenerator(4, 12, ctx)
    monkeypatch.setattr(generator._clients, "ensure_bm25_loaded", AsyncMock(return_value=None))
    task = asyncio.create_task(generator.generate({"question": "q"}, ctx))
    try:
        await asyncio.to_thread(started.wait, 5)
        assert started.is_set()
        for _ in range(cancel_count):
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not task.done()
            assert not finished.is_set()
            llm.close.assert_not_awaited()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await generator.close()
    llm.close.assert_awaited_once()
    assert finished.is_set()
