"""Worker startup and durable sweep progress/failure regression coverage."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from domain.entities.benchmark_sweep import BenchmarkSweep
from domain.value_objects.sweep_status import BenchmarkSweepStatus
from infrastructure.benchmark.sweep_strategies import EnumeratedSweepStrategy
from infrastructure.worker import sweep
from presentation.cli.commands import worker


@pytest.mark.asyncio
async def test_worker_initializes_breakers_after_config_resync(monkeypatch):
    from composition import container as container_module
    from infrastructure.database.database import database
    from infrastructure.redis.redis_client import redis_client
    from infrastructure.resilience import circuit_breaker

    listener = SimpleNamespace(resync=AsyncMock(), start=AsyncMock())
    container = SimpleNamespace(
        init=Mock(),
        infrastructure=SimpleNamespace(
            domain_registry=None,
            events=SimpleNamespace(config_listener=listener),
        ),
    )
    monkeypatch.setattr(container_module, "Container", Mock(return_value=container))
    monkeypatch.setattr(database, "connect", AsyncMock())
    monkeypatch.setattr(redis_client, "init", AsyncMock())
    monkeypatch.setattr(circuit_breaker, "_breakers", {})
    monkeypatch.setattr(worker.settings, "llm_breaker_fail_max", 7)
    monkeypatch.setattr(worker.settings, "llm_breaker_timeout_duration", 19)

    async def resync(**kwargs):
        # Simulate database settings replacing the startup defaults.
        worker.settings.llm_breaker_fail_max = 9

    listener.resync.side_effect = resync

    async def start():
        for name in ("llm_generate", "llm_auxiliary"):
            breaker = circuit_breaker.get_breaker(name)
            assert breaker._fail_max == 9
            assert breaker._timeout_duration == 19

    listener.start.side_effect = start
    ctx = {}
    await worker.on_startup(ctx)
    assert ctx["container"] is container
    listener.start.assert_awaited_once()


class SweepFactory:
    def __init__(self):
        self.saved = BenchmarkSweep(id=1)
        self.sweeps = SimpleNamespace(
            get_by_id=AsyncMock(return_value=self.saved),
            update_status=AsyncMock(),
            update_progress=AsyncMock(),
        )
        self.sweeps.set_best_run = AsyncMock()
        self.runs = SimpleNamespace(save_for_sweep=AsyncMock(return_value=SimpleNamespace(id=1)))
        self.jobs = SimpleNamespace(
            mark_running=AsyncMock(),
            mark_done=AsyncMock(),
            mark_failed=AsyncMock(),
            touch_heartbeat=AsyncMock(),
        )
        self.committed_progress = []

    def create(self, master=False):
        return SweepUow(self)


class SweepUow:
    def __init__(self, factory):
        self.factory = factory
        self.benchmark_sweeps = factory.sweeps
        self.background_jobs = factory.jobs
        self.benchmark_runs = factory.runs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        if not args[0]:
            self.factory.committed_progress = list(self.benchmark_sweeps.update_progress.await_args_list)


def sweep_context(factory, attempt=1):
    return {
        "container": SimpleNamespace(infrastructure=SimpleNamespace(db=SimpleNamespace(uow_factory=factory))),
        "job_try": attempt,
    }


@pytest.mark.asyncio
async def test_progress_is_committed_before_publication_and_done(monkeypatch):
    factory = SweepFactory()
    events = []

    async def publish(sweep_id, event):
        if "evaluated" in event:
            assert factory.committed_progress[-1].args == (1, event["evaluated"], event["total"])
        events.append(event)

    async def run_sweep(**kwargs):
        callback = kwargs["progress_callback"]
        await callback(1, 4, {})
        await callback(0, 2, {"phase": "full_evaluation"})
        await callback(1, 2, {"phase": "full_evaluation"})
        return []

    engine = SimpleNamespace(run_sweep=run_sweep)
    monkeypatch.setattr(sweep, "create_sweep_engine", lambda infra: engine)
    monkeypatch.setattr(sweep, "publish_sweep_event", publish)
    await sweep.run_sweep_task(sweep_context(factory), sweep_id=1, job_id=2)
    assert [e.get("evaluated") for e in events[:-1]] == [1, 0, 1]
    assert events[-1]["done"] is True
    factory.sweeps.update_status.assert_awaited_with(1, BenchmarkSweepStatus.DONE.value)
    factory.jobs.mark_done.assert_awaited_once_with(2)


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt", [1, 3])
async def test_failed_sweep_retries_then_records_error_and_notifies(monkeypatch, attempt):
    factory = SweepFactory()
    engine = SimpleNamespace(run_sweep=AsyncMock(side_effect=KeyError("llm_auxiliary missing")))
    publish = AsyncMock()
    monkeypatch.setattr(sweep, "create_sweep_engine", lambda infra: engine)
    monkeypatch.setattr(sweep, "publish_sweep_event", publish)
    await sweep.run_sweep_task(sweep_context(factory, attempt), sweep_id=1, job_id=2)
    factory.sweeps.update_status.assert_awaited_with(1, BenchmarkSweepStatus.FAILED.value)
    assert "llm_auxiliary missing" in factory.jobs.mark_failed.await_args.args[1]
    assert publish.await_args.args[1]["done"] is True
    assert "llm_auxiliary missing" in publish.await_args.args[1]["error"]
    factory.jobs.mark_done.assert_not_called()


@pytest.mark.asyncio
async def test_timeout_marks_sweep_and_job_failed(monkeypatch):
    factory = SweepFactory()
    engine = SimpleNamespace(run_sweep=AsyncMock(side_effect=asyncio.CancelledError()))
    monkeypatch.setattr(sweep, "create_sweep_engine", lambda infra: engine)
    monkeypatch.setattr(sweep, "publish_sweep_event", AsyncMock())
    with pytest.raises(asyncio.CancelledError):
        await sweep.run_sweep_task(sweep_context(factory), sweep_id=1, job_id=2)
    factory.sweeps.update_status.assert_awaited_with(1, BenchmarkSweepStatus.FAILED.value)
    factory.jobs.mark_failed.assert_awaited_once()


@pytest.mark.asyncio
async def test_strategy_awaits_progress_before_next_score(monkeypatch):
    scores = []
    progress = []

    async def to_thread(function, *args):
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", to_thread)

    def score(config):
        assert len(progress) == len(scores)
        scores.append(config)
        return {"composite_score": 1}

    async def callback(evaluated, total, result):
        await asyncio.sleep(0)
        progress.append((evaluated, total))

    await EnumeratedSweepStrategy([{"top_k": 1}, {"top_k": 2}]).evaluate(score, callback, None)
    assert progress == [(1, 2), (2, 2)]


@pytest.mark.asyncio
async def test_partial_judge_coverage_is_failed_and_resumable_without_traceback(monkeypatch, caplog):
    factory = SweepFactory()
    result = {
        "config": {},
        "llm_evaluated": True,
        "evaluation_complete": False,
        "full_metrics": {"total_questions": 30, "judge_evaluated_count": 27, "judge_error_count": 3},
    }
    engine = SimpleNamespace(run_sweep=AsyncMock(return_value=[result]))
    monkeypatch.setattr(sweep, "create_sweep_engine", lambda infra: engine)
    monkeypatch.setattr(sweep, "publish_sweep_event", AsyncMock())
    await sweep.run_sweep_task(sweep_context(factory), sweep_id=1, job_id=2)
    factory.sweeps.update_status.assert_awaited_with(1, BenchmarkSweepStatus.FAILED.value)
    factory.jobs.mark_done.assert_not_awaited()
    assert "27 из 30" in factory.jobs.mark_failed.await_args.args[1]
    records = [record for record in caplog.records if "incomplete" in record.message]
    assert records and all(record.exc_info is None for record in records)
    assert sweep.publish_sweep_event.await_args.args[1]["error"]
