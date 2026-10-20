"""Benchmark repositories: domain models and rules."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from domain.benchmark.question import BenchmarkQuestion
from domain.benchmark.run import BenchmarkRun
from domain.benchmark.sweep import BenchmarkSweep


@runtime_checkable
class BenchmarkQuestionRepository(Protocol):
    async def get_by_id(self, question_id: int) -> BenchmarkQuestion | None: ...

    async def list_active(self, *, dataset: str) -> list[BenchmarkQuestion]:
        """Read the complete active dataset in one database snapshot."""
        ...

    async def list_items(
        self,
        *,
        dataset: str | None = None,
        tag: str | None = None,
        search: str | None = None,
        is_active: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[BenchmarkQuestion]: ...

    async def count(
        self,
        *,
        dataset: str | None = None,
        tag: str | None = None,
        search: str | None = None,
        is_active: bool | None = None,
    ) -> int: ...

    async def create(self, question: BenchmarkQuestion) -> BenchmarkQuestion: ...

    async def update(self, question_id: int, **fields) -> BenchmarkQuestion | None: ...

    async def delete(self, question_id: int) -> bool: ...

    async def bulk_create(self, questions: list[BenchmarkQuestion]) -> int: ...

    async def get_datasets(self) -> list[str]: ...

    async def count_by_dataset(self) -> dict[str, int]: ...


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


@runtime_checkable
class BenchmarkSweepRepository(Protocol):
    async def get_by_id(self, sweep_id: int, *, for_update: bool = False) -> BenchmarkSweep | None: ...

    async def create(self, sweep: BenchmarkSweep) -> BenchmarkSweep: ...

    async def update_status(self, sweep_id: int, status: str) -> None: ...

    async def fail_dead_jobs(self) -> int: ...

    async def requeue(self, sweep_id: int) -> bool: ...

    async def set_job_id(self, sweep_id: int, job_id: int) -> None: ...

    async def update_progress(self, sweep_id: int, evaluated: int, total: int) -> None: ...

    async def increment_evaluated(self, sweep_id: int) -> None: ...

    async def set_best_run(self, sweep_id: int, run_id: int) -> None: ...

    async def list_items(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[BenchmarkSweep]: ...

    async def count(self) -> int: ...

    async def has_active(self) -> bool: ...

    async def load_checkpoint(self, sweep_id: int, key: str) -> object | None: ...

    async def save_checkpoint(self, sweep_id: int, key: str, value: object) -> None: ...

    async def delete_checkpoints(self, sweep_id: int) -> None: ...

    async def cleanup_checkpoints(self, cutoff: datetime) -> int: ...
