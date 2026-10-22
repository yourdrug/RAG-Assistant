"""Durable database checkpoints and deterministic benchmark identities."""

from application.benchmark.identity import fingerprint


__all__ = ["fingerprint"]
from pathlib import Path
from typing import cast

from application.benchmark import BenchmarkCheckpoints
from infrastructure.benchmark.judging.judge_rubric import JUDGE_RUBRIC_VERSION
from infrastructure.benchmark.judging.prompts import INSTRUCTIONS


class DatabaseBenchmarkCheckpoints:
    def __init__(self, uow_factory, sweep_id: int) -> None:
        self._uow_factory = uow_factory
        self._sweep_id = sweep_id

    async def load(self, key: str) -> object | None:
        async with self._uow_factory.create(master=True) as uow:
            return await uow.benchmark_sweeps.load_checkpoint(self._sweep_id, key)

    async def save(self, key: str, value: object) -> None:
        async with self._uow_factory.create(master=True) as uow:
            await uow.benchmark_sweeps.save_checkpoint(self._sweep_id, key, value)


def refresh_judge_rubric(stages: dict) -> bool:
    """Rejudge old checkpoints while preserving generated answers and evidence."""
    if stages.get("judge_rubric_version") == JUDGE_RUBRIC_VERSION:
        return False

    for key in ("grouped_metrics", "generator_metrics", "context_metrics", "evidence_judge"):
        stages.pop(key, None)

    stages["judge_rubric_version"] = JUDGE_RUBRIC_VERSION
    return True


def restore_grouped_scores(stages: dict) -> dict:

    saved_scores = dict(stages.get("grouped_metrics", {}))

    # Import successes from older per-metric checkpoints on first resume.
    for group in ("generator_metrics", "context_metrics"):
        values = stages.get(group, {})

        for key in INSTRUCTIONS:
            if values.get(key) is not None:
                saved_scores.setdefault(
                    key, {"score": values[key], "reason": values.get(f"{key}_reason", "")}
                )

    legacy = stages.get("evidence_judge", {})

    for key, value in legacy.get("scores", {}).items():
        if value is not None:
            saved_scores.setdefault(
                key,
                {"score": value, "reason": legacy.get("details", {}).get(key, {}).get("reason", "")},
            )

    return saved_scores


async def load_stages(checkpoints: BenchmarkCheckpoints | None, path: Path | None) -> dict:
    if path is None:
        return {}

    if checkpoints is None:
        message = "A database checkpoint store is required for stage recovery"
        raise RuntimeError(message)

    return cast("dict", await checkpoints.load(str(path)) or {})


async def save_stages(checkpoints: BenchmarkCheckpoints | None, path: Path | None, stages: dict) -> None:
    if path is not None:
        if checkpoints is None:
            message = "A database checkpoint store is required for stage recovery"
            raise RuntimeError(message)

        await checkpoints.save(str(path), stages)
