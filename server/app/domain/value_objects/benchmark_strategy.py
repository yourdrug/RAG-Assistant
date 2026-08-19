"""Benchmark sweep strategy."""

from __future__ import annotations

from enum import StrEnum


class BenchmarkStrategy(StrEnum):
    GRID = "grid"
    RANDOM = "random"
    SUCCESSIVE_HALVING = "successive_halving"
