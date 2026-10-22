"""Periodic retention of database recovery data, serialized with sweep resume."""

from datetime import UTC, datetime, timedelta


CHECKPOINT_RETENTION = timedelta(days=1)


async def cron_sweep_checkpoint_cleanup(ctx: dict) -> None:
    factory = ctx["container"].infrastructure.db.uow_factory
    async with factory.create(master=True) as uow:
        await uow.benchmark_sweeps.cleanup_checkpoints(datetime.now(UTC) - CHECKPOINT_RETENTION)
