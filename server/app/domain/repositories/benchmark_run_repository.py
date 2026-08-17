"""BenchmarkRunRepository — persistence protocol for benchmark run results."""

from __future__ import annotations

from typing import Protocol

from domain.entities.benchmark_run import BenchmarkRun


class BenchmarkRunRepository(Protocol):
    async def get_by_id(self, run_id: int) -> BenchmarkRun | None: ...

    async def create(self, run: BenchmarkRun) -> BenchmarkRun: ...

    async def list(
        self,
        *,
        sweep_id: int | None = None,
        dataset: str | None = None,
        sort_by: str = "creation_date",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> list[BenchmarkRun]: ...

    async def count(
        self,
        *,
        sweep_id: int | None = None,
        dataset: str | None = None,
    ) -> int: ...

    async def get_by_ids(self, ids: list[int]) -> list[BenchmarkRun]: ...

    async def get_latest(self, dataset: str | None = None) -> BenchmarkRun | None: ...
