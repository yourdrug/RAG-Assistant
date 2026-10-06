"""Benchmark sweep strategy."""

from __future__ import annotations

from enum import StrEnum


class BenchmarkStrategy(StrEnum):
    GRID = "grid"
    RANDOM = "random"
    OPTUNA_TPE = "optuna_tpe"
    # Deprecated input/persisted value; executes the same TPE search.
    SUCCESSIVE_HALVING = "successive_halving"
