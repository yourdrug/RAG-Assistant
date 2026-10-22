"""Retention boundaries, reports preservation and cleanup after committed success."""

from unittest.mock import AsyncMock

import pytest

from domain.benchmark.sweep import BenchmarkSweepStatus
from infrastructure.worker import checkpoint_cleanup as cron
from infrastructure.worker import sweep as worker
from test_worker_sweep_lifecycle import SweepFactory, sweep_context


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
