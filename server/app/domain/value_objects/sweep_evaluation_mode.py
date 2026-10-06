"""Scope of the expensive evaluation in a parameter sweep."""

from enum import StrEnum

from domain.value_objects.benchmark_strategy import BenchmarkStrategy


class SweepEvaluationMode(StrEnum):
    FAST = "fast"
    FULL = "full"


def validate_sweep_mode(mode: str, strategy: str) -> SweepEvaluationMode:
    resolved = SweepEvaluationMode(mode)
    if resolved == SweepEvaluationMode.FULL and strategy != BenchmarkStrategy.GRID:
        raise ValueError("Full evaluation requires grid search to evaluate every combination")
    return resolved
