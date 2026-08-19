"""Application service for benchmark result listing and detail retrieval.

Reads from BenchmarkRunRepository (DB) instead of JSON files on disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from application.ports.unit_of_work_factory import UnitOfWorkFactory

logger = logging.getLogger("default")


@dataclass(frozen=True)
class BenchmarkResultSummary:
    id: int
    config_json: dict = field(default_factory=dict)
    summary_metrics: dict = field(default_factory=dict)
    duration_sec: float = 0.0
    llm_evaluated: bool = False
    dataset: str = "main"
    sweep_id: int | None = None
    creation_date: object = None


@dataclass(frozen=True)
class BenchmarkResultDetail:
    id: int
    summary: BenchmarkResultSummary
    per_question_results: dict | None = None


@dataclass(frozen=True)
class BenchmarkResultsList:
    results: list[BenchmarkResultSummary] = field(default_factory=list)
    total: int = 0


class BenchmarkResultService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def list_results(
        self,
        dataset: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> BenchmarkResultsList:
        async with self._uow_factory.create() as uow:
            runs = await uow.benchmark_runs.list(
                dataset=dataset,
                limit=limit,
                offset=offset,
            )
            total = await uow.benchmark_runs.count(dataset=dataset)

        summaries = [
            BenchmarkResultSummary(
                id=r.id,
                config_json=r.config_json,
                summary_metrics=r.summary_metrics,
                duration_sec=r.duration_sec,
                llm_evaluated=r.llm_evaluated,
                dataset=r.dataset,
                sweep_id=r.sweep_id,
                creation_date=r.creation_date,
            )
            for r in runs
        ]
        return BenchmarkResultsList(results=summaries, total=total)

    async def get_result(self, run_id: int) -> BenchmarkResultDetail | None:
        async with self._uow_factory.create() as uow:
            run = await uow.benchmark_runs.get_by_id(run_id)

        if run is None:
            return None

        summary = BenchmarkResultSummary(
            id=run.id,
            config_json=run.config_json,
            summary_metrics=run.summary_metrics,
            duration_sec=run.duration_sec,
            llm_evaluated=run.llm_evaluated,
            dataset=run.dataset,
            sweep_id=run.sweep_id,
            creation_date=run.creation_date,
        )
        return BenchmarkResultDetail(
            id=run.id,
            summary=summary,
            per_question_results=run.per_question_results,
        )
