"""Periodic retention of database recovery data, serialized with sweep resume."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any


CHECKPOINT_RETENTION = timedelta(days=1)


async def cron_sweep_checkpoint_cleanup(ctx: dict[str, Any]) -> None:
    factory = ctx["uow_factory"]
    async with factory.create(master=True) as uow:
        await uow.benchmark_sweeps.cleanup_checkpoints(datetime.now(UTC) - CHECKPOINT_RETENTION)
