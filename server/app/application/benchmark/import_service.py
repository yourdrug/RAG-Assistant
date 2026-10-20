"""Idempotent import of validated legacy runs into the existing run repository."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from application.uow import UnitOfWorkFactory
    from domain.benchmark.run import BenchmarkRun


from application.benchmark.dto import BenchmarkImportResult


class BenchmarkImportService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self.uow_factory = uow_factory

    async def import_runs(self, runs: list[BenchmarkRun], *, apply: bool = False) -> BenchmarkImportResult:
        imported = duplicates = 0
        seen: set[str] = set()

        for run in runs:
            if run.legacy_hash is None:
                message = "Legacy provenance hash is required"
                raise ValueError(message)

            async with self.uow_factory.create(master=True) as uow:
                existing = await uow.benchmark_runs.get_by_legacy_hash(run.legacy_hash)

                if existing is not None or run.legacy_hash in seen:
                    duplicates += 1
                elif apply:
                    await uow.benchmark_runs.save_once(run)
                    imported += 1
                else:
                    imported += 1
            seen.add(run.legacy_hash)

        return BenchmarkImportResult(imported, duplicates, dry_run=not apply)
