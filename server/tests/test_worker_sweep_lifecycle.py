"""Worker startup and durable sweep progress/failure regression coverage."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self, cast
from unittest.mock import AsyncMock, Mock

import pytest

from application.benchmark.services import BenchmarkSweepService
from application.benchmark.sweep_jobs import SweepJobService
from composition import worker as worker_composition
from composition.cli.commands import worker
from config import settings
from domain.benchmark.judge import judge_coverage
from domain.benchmark.sweep import BenchmarkSweep, BenchmarkSweepStatus
from infrastructure.benchmark.sweep.strategies import EnumeratedSweepStrategy
from infrastructure.database.database import database
from infrastructure.redis.redis_client import redis_client
from infrastructure.resilience import circuit_breaker
from infrastructure.worker import sweep
from tests.fakes import require_present


if TYPE_CHECKING:
    from application.uow.factory import UnitOfWorkFactory
    from composition.application import ApplicationContainer
    from composition.infrastructure import InfrastructureContainer
    from domain.identity.roles import UserRole


@pytest.mark.asyncio
async def test_worker_initializes_breakers_after_config_resync(monkeypatch: pytest.MonkeyPatch) -> None:

    listener = SimpleNamespace(resync=AsyncMock(), start=AsyncMock())
    container = SimpleNamespace(
        init=Mock(),
        infrastructure=SimpleNamespace(
            domain_registry=None,
            bootstrap_configuration=AsyncMock(),
            events=SimpleNamespace(config_listener=listener),
        ),
    )
    monkeypatch.setattr(worker, "Container", Mock(return_value=container))
    dependencies = {"sweep_engine": object()}

    def assemble() -> Any:
        container.infrastructure.bootstrap_configuration.assert_awaited_once_with(role="worker")
        return dependencies

    container.worker_dependencies = assemble
    monkeypatch.setattr(database, "connect", AsyncMock())
    monkeypatch.setattr(redis_client, "init", AsyncMock())
    monkeypatch.setattr(circuit_breaker, "_breakers", {})
    monkeypatch.setattr(settings, "llm_breaker_fail_max", 7)
    monkeypatch.setattr(settings, "llm_breaker_timeout_duration", 19)

    async def resync(*, role: UserRole) -> None:
        assert role == "worker"
        # Simulate database settings replacing the startup defaults.
        settings.llm_breaker_fail_max = 9

    container.infrastructure.bootstrap_configuration.side_effect = resync

    async def start() -> None:
        for name in ("llm_generate", "llm_auxiliary"):
            breaker = circuit_breaker.get_breaker(name)
            assert breaker._fail_max == 9
            assert breaker._timeout_duration == 19

    listener.start.side_effect = start
    ctx: dict[str, Any] = {}
    await worker.on_startup(ctx)
    assert ctx["container"] is container
    assert ctx["sweep_engine"] is dependencies["sweep_engine"]
    ctx["config_resync_task"].cancel()
    await asyncio.gather(ctx["config_resync_task"], return_exceptions=True)


class SweepFactory:
    def __init__(self) -> None:
        self.saved = BenchmarkSweep(id=1)
        self.sweeps = SimpleNamespace(
            get_by_id=AsyncMock(return_value=self.saved),
            update_status=AsyncMock(),
            update_progress=AsyncMock(),
            delete_checkpoints=AsyncMock(),
            load_checkpoint=AsyncMock(return_value=None),
            save_checkpoint=AsyncMock(),
        )
        self.sweeps.set_best_run = AsyncMock()
        self.runs = SimpleNamespace(save_for_sweep=AsyncMock(return_value=SimpleNamespace(id=1)))
        self.jobs = SimpleNamespace(
            mark_running=AsyncMock(),
            mark_done=AsyncMock(),
            mark_failed=AsyncMock(),
            touch_heartbeat=AsyncMock(),
        )
        self.committed_progress: list[Any] = []

    def create(self, master: bool = False) -> SweepUow:
        _ = master
        return SweepUow(self)


class SweepUow:
    def __init__(self, factory: SweepFactory) -> None:
        self.factory = factory
        self.benchmark_sweeps = factory.sweeps
        self.background_jobs = factory.jobs
        self.benchmark_runs = factory.runs

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        if not args[0]:
            self.factory.committed_progress = list(self.benchmark_sweeps.update_progress.await_args_list)


@pytest.mark.asyncio
async def test_progress_snapshot_survives_late_observers_without_answer_content() -> None:
    factory = SweepFactory()
    factory.saved.job_id = 2
    factory.saved.status = BenchmarkSweepStatus.RUNNING
    service = SweepJobService(cast("UnitOfWorkFactory", factory), AsyncMock(), AsyncMock())
    result = {"config": {"top_k": 3}, "avg_hit_rate": 0.5, "results": [{"answer": "private"}]}
    await service.record_sweep_progress(factory.saved, 1, 2, 4, result)
    snapshot = factory.sweeps.save_checkpoint.await_args.args[2]
    assert snapshot["phase"] == "retrieval"
    assert snapshot["evaluated"] == 2
    assert snapshot["total"] == 4
    assert snapshot["latest"] == {"config": {"top_k": 3}, "avg_hit_rate": 0.5}
    assert snapshot["updated_at"]
    factory.sweeps.load_checkpoint.return_value = snapshot
    reader = BenchmarkSweepService(cast("UnitOfWorkFactory", factory))
    assert await reader.progress(1) == snapshot
    factory.saved.job_id = 3
    assert await reader.progress(1) is None


@pytest.mark.asyncio
async def test_progress_snapshot_records_phase_reset() -> None:
    factory = SweepFactory()
    service = SweepJobService(cast("UnitOfWorkFactory", factory), AsyncMock(), AsyncMock())
    await service.record_sweep_progress(factory.saved, 1, 0, 2, {"phase": "full_evaluation"})
    snapshot = factory.sweeps.save_checkpoint.await_args.args[2]
    assert snapshot["phase"] == "full_evaluation"
    assert snapshot["evaluated"] == 0
    assert snapshot["total"] == 2
    assert snapshot["latest"] == {}


@pytest.mark.asyncio
async def test_sweep_saves_answer_coverage_in_history_summary() -> None:
    factory = SweepFactory()
    best = {
        "llm_evaluated": True,
        "full_metrics": {
            "avg_answer_condition_coverage": 0.6667,
            "avg_answer_fact_coverage": 0.8,
            "answer_condition_coverage_evaluated_count": 6,
        },
    }

    await SweepJobService(cast("UnitOfWorkFactory", factory), AsyncMock(), AsyncMock()).save_sweep_run(
        factory.saved, 1, best
    )

    saved = factory.runs.save_for_sweep.await_args.args[0]
    assert saved.summary_metrics["avg_answer_condition_coverage"] == 0.6667
    assert saved.summary_metrics["avg_answer_fact_coverage"] == 0.8
    assert saved.summary_metrics["answer_condition_coverage_evaluated_count"] == 6


def sweep_context(factory: SweepFactory, attempt: int = 1, *, engine: Any = None) -> dict[str, Any]:
    return {
        "uow_factory": factory,
        "job_try": attempt,
        "sweep_engine": engine,
        "sweep_jobs": SweepJobService(
            cast("UnitOfWorkFactory", factory), sweep.publish_sweep_event, AsyncMock()
        ),
    }


@pytest.mark.asyncio
async def test_progress_is_committed_before_publication_and_done(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = SweepFactory()
    events = []

    async def publish(sweep_id: int, event: Any) -> None:
        if "evaluated" in event:
            assert factory.committed_progress[-1].args == (1, event["evaluated"], event["total"])

        events.append(event)

    async def run_sweep(**kwargs: Any) -> list[Any]:
        callback = kwargs["progress_callback"]
        await callback(1, 4, {})
        await callback(0, 2, {"phase": "full_evaluation"})
        await callback(1, 2, {"phase": "full_evaluation"})
        return []

    engine = SimpleNamespace(run_sweep=run_sweep)
    monkeypatch.setattr(sweep, "publish_sweep_event", publish)
    await sweep.run_sweep_task(sweep_context(factory, engine=engine), sweep_id=1, job_id=2)
    assert [e.get("evaluated") for e in events[:-1]] == [1, 0, 1]
    assert events[-1]["done"] is True
    factory.sweeps.update_status.assert_awaited_with(1, BenchmarkSweepStatus.DONE.value)
    factory.jobs.mark_done.assert_awaited_once_with(2)


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt", [1, 3])
async def test_failed_sweep_retries_then_records_error_and_notifies(
    monkeypatch: pytest.MonkeyPatch, attempt: int
) -> None:
    factory = SweepFactory()
    engine = SimpleNamespace(run_sweep=AsyncMock(side_effect=KeyError("llm_auxiliary missing")))
    publish = AsyncMock()
    monkeypatch.setattr(sweep, "publish_sweep_event", publish)
    await sweep.run_sweep_task(sweep_context(factory, attempt, engine=engine), sweep_id=1, job_id=2)
    factory.sweeps.update_status.assert_awaited_with(1, BenchmarkSweepStatus.FAILED.value)
    assert "llm_auxiliary missing" in factory.jobs.mark_failed.await_args.args[1]
    assert require_present(publish.await_args).args[1]["done"] is True
    assert "llm_auxiliary missing" in require_present(publish.await_args).args[1]["error"]
    factory.jobs.mark_done.assert_not_called()


@pytest.mark.asyncio
async def test_empty_timeout_message_is_saved_and_published(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = SweepFactory()
    engine = SimpleNamespace(run_sweep=AsyncMock(side_effect=TimeoutError()))
    publish = AsyncMock()
    monkeypatch.setattr(sweep, "publish_sweep_event", publish)

    await sweep.run_sweep_task(sweep_context(factory, engine=engine), sweep_id=1, job_id=2)

    factory.sweeps.update_status.assert_awaited_with(1, BenchmarkSweepStatus.FAILED.value)
    factory.jobs.mark_failed.assert_awaited_once_with(2, "TimeoutError")
    assert require_present(publish.await_args).args[1]["error"] == "TimeoutError"
    factory.jobs.mark_done.assert_not_called()


@pytest.mark.asyncio
async def test_timeout_marks_sweep_and_job_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = SweepFactory()
    engine = SimpleNamespace(run_sweep=AsyncMock(side_effect=asyncio.CancelledError()))
    monkeypatch.setattr(sweep, "publish_sweep_event", AsyncMock())
    with pytest.raises(asyncio.CancelledError):
        await sweep.run_sweep_task(sweep_context(factory, engine=engine), sweep_id=1, job_id=2)
    factory.sweeps.update_status.assert_awaited_with(1, BenchmarkSweepStatus.FAILED.value)
    factory.jobs.mark_failed.assert_awaited_once()


@pytest.mark.asyncio
async def test_strategy_awaits_progress_before_next_score(monkeypatch: pytest.MonkeyPatch) -> None:
    scores: list[Any] = []
    progress: list[Any] = []

    async def to_thread(function: Any, *args: Any) -> Any:
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", to_thread)

    def score(config: dict[str, Any]) -> dict[str, Any]:
        assert len(progress) == len(scores)
        scores.append(config)
        return {"composite_score": 1}

    async def callback(evaluated: Any, total: Any, result: object) -> None:
        await asyncio.sleep(0)
        progress.append((evaluated, total))

    await EnumeratedSweepStrategy([{"top_k": 1}, {"top_k": 2}]).evaluate(score, callback, None)
    assert progress == [(1, 2), (2, 2)]


@pytest.mark.asyncio
async def test_partial_judge_coverage_is_failed_and_resumable_without_traceback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    factory = SweepFactory()
    result = {
        "config": {},
        "llm_evaluated": True,
        "evaluation_complete": False,
        "full_metrics": {"total_questions": 30, "judge_evaluated_count": 27, "judge_error_count": 3},
    }
    engine = SimpleNamespace(run_sweep=AsyncMock(return_value=[result]))
    monkeypatch.setattr(sweep, "publish_sweep_event", AsyncMock())
    await sweep.run_sweep_task(sweep_context(factory, engine=engine), sweep_id=1, job_id=2)
    factory.sweeps.update_status.assert_awaited_with(1, BenchmarkSweepStatus.FAILED.value)
    factory.jobs.mark_done.assert_not_awaited()
    assert "27 из 30" in factory.jobs.mark_failed.await_args.args[1]
    records = [record for record in caplog.records if "incomplete" in record.message]
    assert records
    assert all(record.exc_info is None for record in records)
    assert require_present(cast("AsyncMock", sweep.publish_sweep_event).await_args).args[1]["error"]


@pytest.mark.asyncio
async def test_invalid_annotations_are_not_reported_as_successful_or_resumable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = SweepFactory()
    coverage = judge_coverage(
        [{"annotation_validation": {"status": "invalid_annotation"}} for _ in range(24)]
        + [{"annotation_validation": {"status": "unvalidated"}} for _ in range(6)]
    )
    assert coverage["judge_evaluated_count"] == 6
    assert coverage["judge_error_count"] == 0
    assert coverage["invalid_annotation_count"] == 24
    result = {
        "config": {},
        "llm_evaluated": True,
        "evaluation_complete": False,
        "full_metrics": {"total_questions": 30, **coverage},
    }
    engine = SimpleNamespace(run_sweep=AsyncMock(return_value=[result]))
    monkeypatch.setattr(sweep, "publish_sweep_event", AsyncMock())

    await sweep.run_sweep_task(sweep_context(factory, engine=engine), sweep_id=1, job_id=2)

    factory.sweeps.update_status.assert_awaited_with(1, BenchmarkSweepStatus.FAILED.value)
    factory.jobs.mark_done.assert_not_awaited()
    factory.sweeps.delete_checkpoints.assert_not_awaited()
    message = factory.jobs.mark_failed.await_args.args[1]
    assert "6 из 30, ошибок 0" in message
    assert "invalid_annotation: 24" in message
    assert "создайте новый sweep" in message
    assert "Resume выполнит" not in message
    assert (
        require_present(cast("AsyncMock", sweep.publish_sweep_event).await_args).args[1]["error"] == message
    )
    saved = factory.runs.save_for_sweep.await_args.args[0]
    assert saved.summary_metrics["invalid_annotation_count"] == 24


def test_judge_coverage_keeps_failures_separate_from_invalid_annotations() -> None:
    coverage = judge_coverage(
        [
            {"generator_metrics": {"faithfulness": 10}},
            {"generator_metrics": {"faithfulness_error": "timeout"}},
            {"annotation_validation": {"status": "invalid_annotation"}},
        ]
    )
    assert coverage["judge_evaluated_count"] == 1
    assert coverage["judge_error_count"] == 1
    assert coverage["invalid_annotation_count"] == 1


def test_worker_dependencies_are_assembled_in_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = object()
    storage = object()
    infra = SimpleNamespace(
        db=SimpleNamespace(uow=factory),
        ml=SimpleNamespace(file_storage=storage, benchmark_service=object(), ml_clients=None),
        services=SimpleNamespace(cache_invalidator=object(), action_logger=object()),
    )
    processor, ingestion, engine = object(), object(), object()

    constructors: dict[str, Mock] = {}

    for name, service in (
        ("create_document_processor", processor),
        ("create_ingestion_service", ingestion),
        ("create_sweep_engine", engine),
    ):
        constructor = Mock(return_value=service)
        constructors[name] = constructor
        monkeypatch.setattr(worker_composition, name, constructor)

    dependencies = worker_composition.worker_dependencies(
        cast("InfrastructureContainer", infra),
        cast("ApplicationContainer", SimpleNamespace(document_command_service=object())),
    )

    assert dependencies["document_processor"] is processor
    assert dependencies["ingestion_service"] is ingestion
    assert dependencies["sweep_engine"] is engine
    assert dependencies["sweep_jobs"].uow_factory is factory
    assert dependencies["sweep_jobs"].publish is sweep.publish_sweep_event
    constructors["create_document_processor"].assert_called_once_with(infra)
    constructors["create_ingestion_service"].assert_called_once_with(infra)
    constructors["create_sweep_engine"].assert_called_once_with(infra)


@pytest.mark.asyncio
async def test_failed_progress_commit_does_not_publish_success(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = SweepFactory()
    publish = AsyncMock()

    class FailingCommit(SweepUow):
        async def __aexit__(self, *args: object) -> None:
            message = "commit failed"
            raise RuntimeError(message)

    monkeypatch.setattr(factory, "create", lambda **_kwargs: FailingCommit(factory))
    service = SweepJobService(cast("UnitOfWorkFactory", factory), publish, AsyncMock())

    with pytest.raises(RuntimeError, match="commit failed"):
        await service.record_sweep_progress(factory.saved, 1, 1, 2, None)

    publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_sweep_notifies_after_commit_and_retains_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = SweepFactory()
    factory.saved.status = BenchmarkSweepStatus.CANCELLED
    committed = []

    class CommitBeforeNotification(SweepUow):
        async def __aexit__(self, *args: object) -> None:
            await super().__aexit__(*args)
            committed.append(True)

    async def publish(sweep_id: int, event: Any) -> None:
        assert committed == [True]
        assert sweep_id == 1
        assert event == {"done": True, "cancelled": True}

    monkeypatch.setattr(factory, "create", lambda **_kwargs: CommitBeforeNotification(factory))
    await SweepJobService(cast("UnitOfWorkFactory", factory), publish, AsyncMock()).complete_sweep(
        1, [], None
    )

    factory.sweeps.update_status.assert_not_awaited()
    factory.sweeps.delete_checkpoints.assert_not_awaited()
