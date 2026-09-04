"""Adapter for benchmark history — bridges infrastructure.ml.benchmark_history to port."""

from __future__ import annotations

from infrastructure.ml.benchmark_history import compare_runs, get_last_baseline, load_history


class BenchmarkHistoryAdapter:
    """Concrete implementation of BenchmarkHistoryPort."""

    def get_last_baseline(self, data_dir: str) -> dict | None:
        return get_last_baseline(data_dir)

    def load_history(self, data_dir: str) -> list[dict]:
        return load_history(data_dir)

    def compare_runs(self, current: dict, baseline: dict) -> dict:
        return compare_runs(current, baseline)
