"""Application service for benchmark result listing and detail retrieval.

Reads from BenchmarkRunRepository (DB) instead of JSON files on disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from application.dto.benchmark_dto import RegressionCheckOutput, RegressionCheckResult
from application.ports.benchmark_history import BenchmarkHistoryPort
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
    creation_date: datetime | None = None


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
            runs = await uow.benchmark_runs.list_items(
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
            if r.id is not None
        ]
        return BenchmarkResultsList(results=summaries, total=total)

    async def get_result(self, run_id: int) -> BenchmarkResultDetail | None:
        async with self._uow_factory.create() as uow:
            run = await uow.benchmark_runs.get_by_id(run_id)

        if run is None:
            return None

        assert run.id is not None
        summary = BenchmarkResultSummary(
            id=run.id or 0,
            config_json=run.config_json,
            summary_metrics=run.summary_metrics,
            duration_sec=run.duration_sec,
            llm_evaluated=run.llm_evaluated,
            dataset=run.dataset,
            sweep_id=run.sweep_id,
            creation_date=run.creation_date,
        )
        return BenchmarkResultDetail(
            id=run.id or 0,
            summary=summary,
            per_question_results=run.per_question_results,
        )

    async def check_regression(
        self,
        run_id: int | None,
        history_port: BenchmarkHistoryPort,
        data_dir: str,
    ) -> RegressionCheckOutput:
        """Check for regression: compare a run (or latest) against the last baseline."""
        baseline = history_port.get_last_baseline(data_dir)
        if baseline is None:
            return RegressionCheckOutput(passed=True, results=[])

        if run_id is not None:
            detail = await self.get_result(run_id)
            if detail is None:
                return RegressionCheckOutput(passed=True, results=[])
            current = {
                "metrics": detail.summary.summary_metrics,
                "config": detail.summary.config_json,
            }
        else:
            history = history_port.load_history(data_dir)
            if len(history) < 2:
                return RegressionCheckOutput(passed=True, results=[])
            current = history[-1]

        result = history_port.compare_runs(current, baseline)
        return RegressionCheckOutput(
            passed=result["passed"],
            results=[
                RegressionCheckResult(
                    metric=r["metric"],
                    baseline=r.get("baseline"),
                    current=r.get("current"),
                    delta=r.get("delta"),
                    threshold=r["threshold"],
                    failed=r["failed"],
                    note=r.get("note"),
                )
                for r in result["results"]
            ],
        )
