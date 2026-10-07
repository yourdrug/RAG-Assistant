"""Periodic recovery-artifact retention with row locks shared with sweep resume."""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from config import settings
from domain.value_objects.job_status import BackgroundJobStatus
from domain.value_objects.sweep_status import BenchmarkSweepStatus
from infrastructure.benchmark.checkpoint_cleanup import (
    CHECKPOINT_RETENTION,
    RETENTION_FILE,
    checkpoint_last_activity,
    remove_sweep_checkpoints,
)

logger = logging.getLogger("default")


def list_checkpoint_roots(data_dir: str) -> list[Path]:
    parent = Path(data_dir) / "benchmark_results" / "sweeps"
    if not parent.is_dir() or parent.is_symlink():
        return []
    return [
        path for path in parent.iterdir() if path.name.isdecimal() and path.is_dir() and not path.is_symlink()
    ]


def inactive_since(root: Path, finished_at: datetime | None) -> datetime | None:
    if finished_at is not None:
        return finished_at if finished_at.tzinfo else finished_at.replace(tzinfo=UTC)
    marker = root / RETENTION_FILE
    if marker.exists():
        return datetime.fromtimestamp(marker.stat().st_mtime, UTC)
    return checkpoint_last_activity(root)


async def cron_sweep_checkpoint_cleanup(ctx: dict) -> None:
    factory = ctx["container"].infrastructure.db.uow_factory
    cutoff = datetime.now(UTC) - CHECKPOINT_RETENTION
    for root in await asyncio.to_thread(list_checkpoint_roots, settings.data_dir):
        try:
            async with factory.create(master=True) as uow:
                # Resume's conditional UPDATE waits for this lock. Never delete
                # files after checking an unlocked, potentially obsolete status.
                sweep = await uow.benchmark_sweeps.get_by_id(int(root.name), for_update=True)
                if sweep is not None and sweep.status in (
                    BenchmarkSweepStatus.PENDING.value,
                    BenchmarkSweepStatus.RUNNING.value,
                ):
                    continue
                job = await uow.background_jobs.get_by_id(sweep.job_id) if sweep and sweep.job_id else None
                if job is not None and job.status in (
                    BackgroundJobStatus.PENDING.value,
                    BackgroundJobStatus.RUNNING.value,
                ):
                    continue
                done = sweep is not None and sweep.status == BenchmarkSweepStatus.DONE.value
                since = await asyncio.to_thread(inactive_since, root, job.finished_at if job else None)
                if done or (since is not None and since <= cutoff):
                    await asyncio.to_thread(remove_sweep_checkpoints, root)
        except OSError:
            logger.exception("Checkpoint cleanup failed for sweep %s; will retry", root.name)
