"""Remove recovery artifacts while retaining benchmark reports."""

import logging
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from infrastructure.benchmark.checkpoint import write_checkpoint

logger = logging.getLogger("default")
CHECKPOINT_RETENTION = timedelta(days=1)
RETENTION_FILE = "retention.json"
DATASET_SNAPSHOT = "dataset.json"


def sweep_checkpoint_root(data_dir: str, sweep_id: int) -> Path:
    return Path(data_dir) / "benchmark_results" / "sweeps" / str(sweep_id)


def checkpoint_artifacts(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        return []
    names = (
        DATASET_SNAPSHOT,
        RETENTION_FILE,
        Path(DATASET_SNAPSHOT).with_suffix(".tmp").name,
        Path(RETENTION_FILE).with_suffix(".tmp").name,
    )
    artifacts = [root / name for name in names if (root / name).exists()]
    for config in root.iterdir():
        if config.is_dir() and not config.is_symlink() and (config / "checkpoints").exists():
            artifacts.append(config / "checkpoints")
    return artifacts


def remove_sweep_checkpoints(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        return
    for artifact in checkpoint_artifacts(root):
        if artifact.is_dir() and not artifact.is_symlink():
            shutil.rmtree(artifact)
        else:
            artifact.unlink(missing_ok=True)
    for config in root.iterdir():
        if config.is_dir() and not config.is_symlink() and not any(config.iterdir()):
            config.rmdir()
    if not any(root.iterdir()):
        root.rmdir()


def mark_checkpoints_inactive(root: Path) -> None:
    if checkpoint_artifacts(root):
        write_checkpoint(root / RETENTION_FILE, {"inactive_since": datetime.now(UTC).isoformat()})


def checkpoint_last_activity(root: Path) -> datetime | None:
    timestamps = []
    for artifact in checkpoint_artifacts(root):
        timestamps.append(artifact.stat().st_mtime)
        if artifact.is_dir() and not artifact.is_symlink():
            timestamps.extend(path.stat().st_mtime for path in artifact.rglob("*") if not path.is_symlink())
    return datetime.fromtimestamp(max(timestamps), UTC) if timestamps else None


def cleanup_sweep_checkpoints(data_dir: str, sweep_id: int, *, successful: bool) -> None:
    """Cleanup failures must not turn a committed successful benchmark into a failed job."""
    try:
        root = sweep_checkpoint_root(data_dir, sweep_id)
        if successful:
            remove_sweep_checkpoints(root)
        else:
            mark_checkpoints_inactive(root)
    except OSError:
        logger.exception("Could not update checkpoint retention for sweep %d; cron will retry", sweep_id)
