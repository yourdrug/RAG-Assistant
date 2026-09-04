"""Application ports for benchmark history — comparison and baseline loading."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BenchmarkHistoryPort(Protocol):
    """Port for benchmark history operations (baseline, comparison)."""

    def get_last_baseline(self, data_dir: str) -> dict | None: ...
    def load_history(self, data_dir: str) -> list[dict]: ...
    def compare_runs(self, current: dict, baseline: dict) -> dict: ...
