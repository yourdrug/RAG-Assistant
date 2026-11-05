"""Tests for short-term reliability fixes (AUDIT.md items 7-13).

- C-4: ``run_tracked_job`` persists job state on success / failure / cancellation,
  and keeps a heartbeat alive while the body runs.
- M-14: cooperative sweep cancellation (``SweepEngine.check_cancelled``)
  and the double-submit guard in ``BenchmarkSweepService.create``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Self, cast

from arq import Retry
import pytest

from application.benchmark.dto import SweepCreateDTO
from tests.fakes import FakeUnitOfWorkFactory


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


from application.benchmark import BenchmarkSweepService
from application.benchmark.sweep_ports import SweepCancelled
from composition.service_providers import build_sweep_engine
from domain.benchmark.sweep import (
    BenchmarkSweepStatus,
)
from domain.shared.exceptions import BusinessRuleViolationError
from infrastructure.worker import runtime as worker_tasks


if TYPE_CHECKING:
    from application.uow.factory import UnitOfWorkFactory


# ---------------------------------------------------------------------------
# Stateful fake for run_tracked_job tests
# ---------------------------------------------------------------------------


class _StatefulJobRepo:
    def __init__(self) -> None:
        self.status: str | None = None
        self.started = False
        self.finished = False
        self.error: str | None = None
        self.heartbeat_touches = 0

    async def mark_running(self, job_id: int) -> None:
        _ = job_id
        self.started = True
        self.status = "running"

    async def mark_done(self, job_id: int) -> None:
        _ = job_id
        self.finished = True
        self.status = "done"

    async def mark_failed(self, job_id: int, error: str) -> None:
        _ = job_id
        self.finished = True
        self.status = "failed"
        self.error = error

    async def touch_heartbeat(self, job_id: int) -> None:
        _ = job_id
        self.heartbeat_touches += 1


class _TrackedUow:
    def __init__(self, repo: _StatefulJobRepo) -> None:
        self.background_jobs = repo

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _Factory:
    def __init__(self, repo: _StatefulJobRepo) -> None:
        self._repo = repo

    def create(self, master: bool = False) -> Any:
        _ = master
        return _TrackedUow(self._repo)


# ---------------------------------------------------------------------------
# C-4: tracked job state machine
# ---------------------------------------------------------------------------


def _run_tracked(repo: Any, body: Any, **kwargs: Any) -> None:
    asyncio.run(
        worker_tasks.run_tracked_job(
            cast("UnitOfWorkFactory", _Factory(repo)), 1, body, description="test", **kwargs
        )
    )


class TestRunTrackedJob:
    def test_success_marks_running_then_done(self) -> None:
        repo = _StatefulJobRepo()

        async def body() -> None:
            assert repo.status == "running", "body must see the job marked running"

        _run_tracked(repo, body)
        assert repo.status == "done"
        assert repo.started
        assert repo.finished

    def test_exception_schedules_retry(self) -> None:
        repo = _StatefulJobRepo()

        async def body() -> None:
            msg = "boom"
            raise RuntimeError(msg)

        with pytest.raises(Retry) as raised:
            _run_tracked(repo, body)

        assert raised.value.defer_score == 5000
        assert repo.status == "running"
        assert repo.started
        assert not repo.finished

    def test_terminal_exception_marks_failed_without_raising(self) -> None:
        repo = _StatefulJobRepo()

        async def body() -> None:
            msg = "boom"
            raise RuntimeError(msg)

        _run_tracked(repo, body, job_try=3, max_tries=3)
        assert repo.status == "failed"
        assert repo.error == "boom"

    def test_cancellation_marks_failed_and_reraises(self) -> None:
        repo = _StatefulJobRepo()

        async def body() -> None:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            _run_tracked(repo, body)
        assert repo.status == "failed"
        assert repo.error is not None
        assert "cancel" in repo.error.lower()

    @pytest.mark.parametrize("failure", [TimeoutError(), RuntimeError("   ")])
    def test_empty_exception_message_persists_exception_type(self, failure: Any) -> None:
        repo = _StatefulJobRepo()

        async def body() -> None:
            raise failure

        _run_tracked(repo, body, max_tries=1)
        assert repo.finished
        assert repo.error == type(failure).__name__

    def test_heartbeat_touches_while_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(worker_tasks, "HEARTBEAT_INTERVAL_SEC", 0.01)
        repo = _StatefulJobRepo()

        async def body() -> None:
            await asyncio.sleep(0.05)

        _run_tracked(repo, body)
        assert repo.heartbeat_touches >= 1, "heartbeat must refresh while the body runs"

    def test_heartbeat_cancelled_after_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(worker_tasks, "HEARTBEAT_INTERVAL_SEC", 3600)
        repo = _StatefulJobRepo()

        async def body() -> None:
            pass

        _run_tracked(repo, body)
        # No failure — the heartbeat task was cancelled cleanly in finally
        assert repo.status == "done"


# ---------------------------------------------------------------------------
# M-14: cooperative cancellation
# ---------------------------------------------------------------------------


class TestSweepCancellation:
    def test_check_cancelled_noop_without_callback(self) -> None:
        engine = build_sweep_engine(uow_factory=FakeUnitOfWorkFactory())
        asyncio.run(engine.check_cancelled(None, "anywhere"))

    def test_check_cancelled_raises_when_requested(self) -> None:
        engine = build_sweep_engine(uow_factory=FakeUnitOfWorkFactory())

        async def cancelled() -> bool:
            return True

        with pytest.raises(SweepCancelled):
            asyncio.run(engine.check_cancelled(cancelled, "phase A"))

    def test_check_cancelled_continues_when_not_requested(self) -> None:
        engine = build_sweep_engine(uow_factory=FakeUnitOfWorkFactory())

        async def not_cancelled() -> bool:
            return False

        asyncio.run(engine.check_cancelled(not_cancelled, "phase A"))


# ---------------------------------------------------------------------------
# M-14: double-submit guard
# ---------------------------------------------------------------------------


class _ActiveSweepRepo:
    """Fake repo reporting an active (pending/running) sweep."""

    async def create(self, entity: Any) -> Any:
        return entity

    async def has_active(self) -> bool:
        return True


class _InactiveSweepRepo:
    async def create(self, entity: Any) -> Any:
        entity.id = 1
        return entity

    async def has_active(self) -> bool:
        return False


class _SweepUow:
    def __init__(self, repo: Any) -> None:
        self.benchmark_sweeps = repo

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _SweepFactory:
    def __init__(self, repo: Any) -> None:
        self._repo = repo

    def create(self, master: bool = False) -> _SweepUow:
        _ = master
        return _SweepUow(self._repo)


class TestSweepDoubleSubmitGuard:
    def test_create_rejected_when_active_sweep_exists(self) -> None:
        service = BenchmarkSweepService(cast("UnitOfWorkFactory", _SweepFactory(_ActiveSweepRepo())))
        with pytest.raises(BusinessRuleViolationError):
            asyncio.run(service.create(SweepCreateDTO(top_n_llm=0, objective_weights={}, dataset="default")))

    def test_create_allowed_when_no_active_sweep(self) -> None:
        service = BenchmarkSweepService(cast("UnitOfWorkFactory", _SweepFactory(_InactiveSweepRepo())))
        sweep = asyncio.run(
            service.create(SweepCreateDTO(top_n_llm=0, objective_weights={}, dataset="default"))
        )
        assert sweep.status == BenchmarkSweepStatus.PENDING.value
