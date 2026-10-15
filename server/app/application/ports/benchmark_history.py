"""Application ports for benchmark history — comparison and baseline loading."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BenchmarkHistoryPort(Protocol):
    """Port for benchmark history operations (baseline, comparison)."""

    def compare_runs(self, current: dict, baseline: dict) -> dict: ...
