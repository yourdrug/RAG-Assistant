"""Tests for short-term reliability fixes (AUDIT.md items 7-13).

- C-4: ``_run_tracked_job`` persists job state on success / failure / cancellation,
  and keeps a heartbeat alive while the body runs.
- M-14: cooperative sweep cancellation (``SweepEngine.check_cancelled``)
  and the double-submit guard in ``BenchmarkSweepService.create``.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest  # noqa: E402
from domain.exceptions import BusinessRuleViolation  # noqa: E402
from domain.value_objects.sweep_status import BenchmarkSweepStatus  # noqa: E402
from domain.value_objects.sweep_evaluation_mode import SweepEvaluationMode  # noqa: E402

from application.services.benchmark_services import BenchmarkSweepService  # noqa: E402
from infrastructure.benchmark.sweep_engine import SweepCancelled, SweepEngine  # noqa: E402
from infrastructure.worker import tasks as worker_tasks  # noqa: E402

# ---------------------------------------------------------------------------
# Stateful fake for _run_tracked_job tests
# ---------------------------------------------------------------------------


class _StatefulJobRepo:
    def __init__(self):
        self.status: str | None = None
        self.started = False
        self.finished = False
        self.error: str | None = None
        self.heartbeat_touches = 0

    async def mark_running(self, job_id: int) -> None:
        self.started = True
        self.status = "running"

    async def mark_done(self, job_id: int) -> None:
        self.finished = True
        self.status = "done"

    async def mark_failed(self, job_id: int, error: str) -> None:
        self.finished = True
        self.status = "failed"
        self.error = error

    async def touch_heartbeat(self, job_id: int) -> None:
        self.heartbeat_touches += 1


class _TrackedUow:
    def __init__(self, repo: _StatefulJobRepo):
        self.background_jobs = repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class _Factory:
    def __init__(self, repo: _StatefulJobRepo):
        self._repo = repo

    def create(self, master: bool = False):
        return _TrackedUow(self._repo)


# ---------------------------------------------------------------------------
# C-4: tracked job state machine
# ---------------------------------------------------------------------------


def _run_tracked(repo, body, **kwargs):
    asyncio.run(worker_tasks._run_tracked_job(_Factory(repo), 1, body, description="test", **kwargs))


class TestRunTrackedJob:
    def test_success_marks_running_then_done(self):
        repo = _StatefulJobRepo()

        async def body():
            assert repo.status == "running", "body must see the job marked running"

        _run_tracked(repo, body)
        assert repo.status == "done"
        assert repo.started and repo.finished

    def test_exception_schedules_retry(self):
        repo = _StatefulJobRepo()

        async def body():
            raise RuntimeError("boom")

        with pytest.raises(worker_tasks.Retry) as raised:
            _run_tracked(repo, body)

        assert raised.value.defer_score == 5000
        assert repo.status == "running"
        assert repo.started and not repo.finished

    def test_terminal_exception_marks_failed_without_raising(self):
        repo = _StatefulJobRepo()

        async def body():
            raise RuntimeError("boom")

        _run_tracked(repo, body, job_try=3, max_tries=3)
        assert repo.status == "failed"
        assert repo.error == "boom"

    def test_cancellation_marks_failed_and_reraises(self):
        repo = _StatefulJobRepo()

        async def body():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            _run_tracked(repo, body)
        assert repo.status == "failed"
        assert repo.error is not None and "cancel" in repo.error.lower()

    def test_heartbeat_touches_while_running(self, monkeypatch):
        monkeypatch.setattr(worker_tasks, "_HEARTBEAT_INTERVAL_SEC", 0.01)
        repo = _StatefulJobRepo()

        async def body():
            await asyncio.sleep(0.05)

        _run_tracked(repo, body)
        assert repo.heartbeat_touches >= 1, "heartbeat must refresh while the body runs"

    def test_heartbeat_cancelled_after_body(self, monkeypatch):
        monkeypatch.setattr(worker_tasks, "_HEARTBEAT_INTERVAL_SEC", 3600)
        repo = _StatefulJobRepo()

        async def body():
            pass

        _run_tracked(repo, body)
        # No failure — the heartbeat task was cancelled cleanly in finally
        assert repo.status == "done"


# ---------------------------------------------------------------------------
# M-14: cooperative cancellation
# ---------------------------------------------------------------------------


class TestSweepCancellation:
    def test_check_cancelled_noop_without_callback(self):
        engine = SweepEngine(uow_factory=None)
        asyncio.run(engine.check_cancelled(None, "anywhere"))

    def test_check_cancelled_raises_when_requested(self):
        engine = SweepEngine(uow_factory=None)

        async def cancelled() -> bool:
            return True

        with pytest.raises(SweepCancelled):
            asyncio.run(engine.check_cancelled(cancelled, "phase A"))

    def test_check_cancelled_continues_when_not_requested(self):
        engine = SweepEngine(uow_factory=None)

        async def not_cancelled() -> bool:
            return False

        asyncio.run(engine.check_cancelled(not_cancelled, "phase A"))


# ---------------------------------------------------------------------------
# M-14: double-submit guard
# ---------------------------------------------------------------------------


class _ActiveSweepRepo:
    """Fake repo reporting an active (pending/running) sweep."""

    async def create(self, entity):
        return entity

    async def has_active(self) -> bool:
        return True


class _InactiveSweepRepo:
    async def create(self, entity):
        entity.id = 1
        return entity

    async def has_active(self) -> bool:
        return False


class _SweepUow:
    def __init__(self, repo):
        self.benchmark_sweeps = repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class _SweepFactory:
    def __init__(self, repo):
        self._repo = repo

    def create(self, master: bool = False):
        return _SweepUow(self._repo)


class _CreateSweepBody:
    evaluation_mode = SweepEvaluationMode.FAST.value
    strategy = "grid"
    search_space = {}
    objective_weights = {}
    dataset = "default"
    top_n_llm = 0
    judge_model = None


class TestSweepDoubleSubmitGuard:
    def test_create_rejected_when_active_sweep_exists(self):
        service = BenchmarkSweepService(_SweepFactory(_ActiveSweepRepo()))
        with pytest.raises(BusinessRuleViolation):
            asyncio.run(service.create(_CreateSweepBody()))

    def test_create_allowed_when_no_active_sweep(self):
        service = BenchmarkSweepService(_SweepFactory(_InactiveSweepRepo()))
        sweep = asyncio.run(service.create(_CreateSweepBody()))
        assert sweep.status == BenchmarkSweepStatus.PENDING.value
