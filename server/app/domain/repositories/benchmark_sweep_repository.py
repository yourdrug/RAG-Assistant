"""BenchmarkSweepRepository — persistence protocol for parameter sweeps."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from domain.entities.benchmark_sweep import BenchmarkSweep


@runtime_checkable
class BenchmarkSweepRepository(Protocol):
    async def get_by_id(self, sweep_id: int) -> BenchmarkSweep | None: ...

    async def create(self, sweep: BenchmarkSweep) -> BenchmarkSweep: ...

    async def update_status(self, sweep_id: int, status: str) -> None: ...

    async def increment_evaluated(self, sweep_id: int) -> None: ...

    async def set_best_run(self, sweep_id: int, run_id: int) -> None: ...

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[BenchmarkSweep]: ...

    async def count(self) -> int: ...
