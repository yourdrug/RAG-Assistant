"""Retention boundaries, reports preservation and cleanup after committed success."""

from unittest.mock import AsyncMock

import pytest

from domain.benchmark.sweep import BenchmarkSweepStatus
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
async def test_web_cleanup_uses_database_without_filesystem(monkeypatch):
    factory = SweepFactory()
    factory.sweeps.cleanup_checkpoints = AsyncMock()
    await cron.cron_sweep_checkpoint_cleanup(sweep_context(factory))
    factory.sweeps.cleanup_checkpoints.assert_awaited_once()


@pytest.mark.asyncio
async def test_successful_worker_cleans_database_only_after_done_commit(monkeypatch):
    factory = SweepFactory()
    factory.sweeps.delete_checkpoints = AsyncMock()
    monkeypatch.setattr(worker, "publish_sweep_event", AsyncMock())
    await worker.complete_sweep(factory, 1, [], None)
    factory.sweeps.update_status.assert_awaited_with(1, BenchmarkSweepStatus.DONE.value)
    factory.sweeps.delete_checkpoints.assert_awaited_once_with(1)


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
