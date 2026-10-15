"""Periodic retention of database recovery data, serialized with sweep resume."""

from datetime import UTC, datetime

from infrastructure.benchmark.checkpoint_cleanup import CHECKPOINT_RETENTION


async def cron_sweep_checkpoint_cleanup(ctx: dict) -> None:
    factory = ctx["container"].infrastructure.db.uow_factory
    async with factory.create(master=True) as uow:
        await uow.benchmark_sweeps.cleanup_checkpoints(datetime.now(UTC) - CHECKPOINT_RETENTION)
