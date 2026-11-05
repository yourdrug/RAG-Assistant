"""Retention boundaries, reports preservation and cleanup after committed success."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import pytest

from application.benchmark.sweep_jobs import SweepJobService
from domain.benchmark.sweep import BenchmarkSweepStatus
from infrastructure.worker import checkpoint_cleanup as cron
from tests.test_worker_sweep_lifecycle import SweepFactory, sweep_context


if TYPE_CHECKING:
    from application.uow.factory import UnitOfWorkFactory


@pytest.mark.asyncio
async def test_web_cleanup_uses_database_without_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = SweepFactory()
    factory.sweeps.cleanup_checkpoints = AsyncMock()
    await cron.cron_sweep_checkpoint_cleanup(sweep_context(factory))
    factory.sweeps.cleanup_checkpoints.assert_awaited_once()


@pytest.mark.asyncio
async def test_successful_worker_cleans_database_only_after_done_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = SweepFactory()
    factory.sweeps.delete_checkpoints = AsyncMock()
    await SweepJobService(cast("UnitOfWorkFactory", factory), AsyncMock(), AsyncMock()).complete_sweep(
        1, [], None
    )
    factory.sweeps.update_status.assert_awaited_with(1, BenchmarkSweepStatus.DONE.value)
    factory.sweeps.delete_checkpoints.assert_awaited_once_with(1)
