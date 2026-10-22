"""Pure metric comparison; runtime history is stored in PostgreSQL."""

from infrastructure.benchmark.reporting.benchmark_history import compare_runs


class BenchmarkHistoryAdapter:
    def compare_runs(self, current: dict, baseline: dict) -> dict:
        return compare_runs(current, baseline)
