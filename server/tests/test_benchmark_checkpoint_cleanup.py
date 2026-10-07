"""Retention boundaries, reports preservation and cleanup after committed success."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
import os

import pytest

from domain.value_objects.job_status import BackgroundJobStatus
from domain.value_objects.sweep_status import BenchmarkSweepStatus
from infrastructure.benchmark.checkpoint_cleanup import (
    cleanup_sweep_checkpoints,
    remove_sweep_checkpoints,
    sweep_checkpoint_root,
)
from infrastructure.worker import checkpoint_cleanup as cron
from infrastructure.worker import sweep as worker
from test_worker_sweep_lifecycle import SweepFactory, sweep_context


def make_artifacts(data_dir):
    root = sweep_checkpoint_root(str(data_dir), 1)
    checkpoints = root / "config" / "checkpoints" / "fingerprint"
    checkpoints.mkdir(parents=True)
    (checkpoints / "1-1.json").write_text("{}")
    (checkpoints / "1-2.stages.json").write_text("{}")
    (root / "dataset.json").write_text("[]")
    (root / "config" / "benchmark_final.json").write_text("[]")
    (root / "config" / "benchmark_final.csv").write_text("id")
    return root


def assert_cleaned(root):
    assert not (root / "dataset.json").exists()
    assert not (root / "config" / "checkpoints").exists()
    assert (root / "config" / "benchmark_final.json").read_text() == "[]"
    assert (root / "config" / "benchmark_final.csv").read_text() == "id"


def test_remove_checkpoints_keeps_reports_and_cleans_abandoned_temporary_files(tmp_path):
    root = make_artifacts(tmp_path)
    (root / "dataset.tmp").write_text("partial")
    remove_sweep_checkpoints(root)
    assert_cleaned(root)
    assert not (root / "dataset.tmp").exists()
    remove_sweep_checkpoints(root)  # idempotent


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [BenchmarkSweepStatus.FAILED, BenchmarkSweepStatus.CANCELLED])
@pytest.mark.parametrize("hours,expired", [(23.99, False), (24, True), (24.01, True)])
async def test_inactive_retention_is_exactly_one_day(tmp_path, monkeypatch, status, hours, expired):
    root = make_artifacts(tmp_path)
    now = datetime(2026, 10, 7, tzinfo=UTC)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(cron, "datetime", Clock)
    monkeypatch.setattr(cron.settings, "data_dir", str(tmp_path))
    factory = SweepFactory()
    factory.saved.status = status.value
    factory.saved.job_id = 2
    factory.jobs.get_by_id = AsyncMock(
        return_value=SimpleNamespace(
            status=BackgroundJobStatus.FAILED.value, finished_at=now - timedelta(hours=hours)
        )
    )
    await cron.cron_sweep_checkpoint_cleanup(sweep_context(factory))
    factory.sweeps.get_by_id.assert_awaited_once_with(1, for_update=True)
    if expired:
        assert_cleaned(root)
    else:
        assert (root / "dataset.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [BenchmarkSweepStatus.PENDING, BenchmarkSweepStatus.RUNNING])
async def test_active_sweep_keeps_old_checkpoints(tmp_path, monkeypatch, status):
    root = make_artifacts(tmp_path)
    for path in root.rglob("*"):
        os.utime(path, (0, 0))
    factory = SweepFactory()
    factory.saved.status = status.value
    monkeypatch.setattr(cron.settings, "data_dir", str(tmp_path))
    await cron.cron_sweep_checkpoint_cleanup(sweep_context(factory))
    assert (root / "dataset.json").exists()


@pytest.mark.asyncio
async def test_cancelled_sweep_with_live_worker_keeps_checkpoints(tmp_path, monkeypatch):
    root = make_artifacts(tmp_path)
    factory = SweepFactory()
    factory.saved.status = BenchmarkSweepStatus.CANCELLED.value
    factory.saved.job_id = 2
    factory.jobs.get_by_id = AsyncMock(return_value=SimpleNamespace(status=BackgroundJobStatus.RUNNING.value))
    monkeypatch.setattr(cron.settings, "data_dir", str(tmp_path))
    await cron.cron_sweep_checkpoint_cleanup(sweep_context(factory))
    assert (root / "dataset.json").exists()


@pytest.mark.asyncio
async def test_successful_worker_cleans_only_after_results_commit(tmp_path, monkeypatch):
    root = make_artifacts(tmp_path)
    factory = SweepFactory()
    engine = SimpleNamespace(run_sweep=AsyncMock(return_value=[]))
    monkeypatch.setattr(worker.settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(worker, "create_sweep_engine", lambda infra: engine)
    monkeypatch.setattr(worker, "publish_sweep_event", AsyncMock())

    def cleanup(data_dir, sweep_id, *, successful):
        factory.sweeps.update_status.assert_awaited_with(1, BenchmarkSweepStatus.DONE.value)
        cleanup_sweep_checkpoints(data_dir, sweep_id, successful=successful)

    monkeypatch.setattr(worker, "cleanup_sweep_checkpoints", cleanup)
    await worker.run_sweep_task(sweep_context(factory), sweep_id=1, job_id=2)
    assert_cleaned(root)


def test_failed_worker_marks_retention_without_removing_answers(tmp_path):
    root = make_artifacts(tmp_path)
    cleanup_sweep_checkpoints(str(tmp_path), 1, successful=False)
    assert (root / "retention.json").exists()
    assert (root / "config" / "checkpoints" / "fingerprint" / "1-2.stages.json").exists()


def test_cleanup_removes_empty_sweep_directory(tmp_path):
    root = sweep_checkpoint_root(str(tmp_path), 1)
    (root / "config" / "checkpoints").mkdir(parents=True)
    (root / "dataset.json").write_text("[]")
    remove_sweep_checkpoints(root)
    assert not root.exists()


def test_cleanup_does_not_follow_sweep_symlink(tmp_path):
    root = make_artifacts(tmp_path)
    linked = tmp_path / "linked"
    linked.symlink_to(root, target_is_directory=True)
    remove_sweep_checkpoints(linked)
    assert (root / "dataset.json").exists()


@pytest.mark.asyncio
async def test_done_sweep_is_cleaned_by_cron_even_before_retention_expires(tmp_path, monkeypatch):
    root = make_artifacts(tmp_path)
    factory = SweepFactory()
    factory.saved.status = BenchmarkSweepStatus.DONE.value
    monkeypatch.setattr(cron.settings, "data_dir", str(tmp_path))
    await cron.cron_sweep_checkpoint_cleanup(sweep_context(factory))
    assert_cleaned(root)


@pytest.mark.asyncio
async def test_removed_sweep_without_job_uses_artifact_age(tmp_path, monkeypatch):
    root = make_artifacts(tmp_path)
    for path in root.rglob("*"):
        os.utime(path, (0, 0))
    factory = SweepFactory()
    factory.sweeps.get_by_id = AsyncMock(return_value=None)
    monkeypatch.setattr(cron.settings, "data_dir", str(tmp_path))
    await cron.cron_sweep_checkpoint_cleanup(sweep_context(factory))
    assert_cleaned(root)
