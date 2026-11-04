"""Pure metric comparison; runtime history is stored in PostgreSQL."""

from typing import Any

from infrastructure.benchmark.reporting.benchmark_history import compare_runs


class BenchmarkHistoryAdapter:
    def compare_runs(self, current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
        return compare_runs(current, baseline)
