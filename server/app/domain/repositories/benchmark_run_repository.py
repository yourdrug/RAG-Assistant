"""BenchmarkRunRepository — persistence protocol for benchmark run results."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from domain.entities.benchmark_run import BenchmarkRun


@runtime_checkable
class BenchmarkRunRepository(Protocol):
    async def get_by_id(self, run_id: int) -> BenchmarkRun | None: ...

    async def get_by_job_id(self, job_id: int) -> BenchmarkRun | None: ...

    async def get_by_legacy_hash(self, digest: str) -> BenchmarkRun | None: ...

    async def save_once(self, run: BenchmarkRun) -> BenchmarkRun: ...

    async def get_baseline(self, current: BenchmarkRun) -> BenchmarkRun | None: ...

    async def export_header(self, run_id: int) -> dict | None: ...

    async def result_page(self, run_id: int, *, offset: int, limit: int) -> list[dict]: ...

    async def get_sweep_run_ids(self, sweep_id: int, limit: int) -> list[int]: ...

    async def set_artifact(self, run_id: int, format: str, metadata: dict) -> None: ...

    async def create(self, run: BenchmarkRun) -> BenchmarkRun: ...

    async def save_for_sweep(self, run: BenchmarkRun) -> BenchmarkRun: ...

    async def list_items(
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
