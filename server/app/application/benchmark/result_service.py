"""Application service for benchmark result listing and detail retrieval.

Reads from BenchmarkRunRepository (DB) instead of JSON files on disk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from application.benchmark.dto import (
    BenchmarkResultDetail,
    BenchmarkResultsList,
    BenchmarkResultSummary,
    RegressionCheckOutput,
    RegressionCheckResult,
)


if TYPE_CHECKING:
    from application.benchmark.ports import BenchmarkHistoryPort
    from application.uow import UnitOfWorkFactory
    from domain.entities.benchmark_run import BenchmarkRun


class BenchmarkResultService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def get_job_result(self, job_id: int) -> BenchmarkRun | None:
        async with self._uow_factory.create(master=True) as uow:
            return await uow.benchmark_runs.get_by_job_id(job_id)

    async def record_result(self, run: BenchmarkRun) -> BenchmarkRun:
        async with self._uow_factory.create(master=True) as uow:
            return await uow.benchmark_runs.save_once(run)

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
                artifacts=r.artifacts,
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

        if run.id is None:
            message = "BenchmarkRun id is None"
            raise RuntimeError(message)

        summary = BenchmarkResultSummary(
            id=run.id or 0,
            config_json=run.config_json,
            artifacts=run.artifacts,
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
    ) -> RegressionCheckOutput:
        """Compare a DB run with its previous compatible baseline."""
        async with self._uow_factory.create() as uow:
            run = (
                await uow.benchmark_runs.get_by_id(run_id)
                if run_id is not None
                else await uow.benchmark_runs.get_latest()
            )

            if run is None:
                return RegressionCheckOutput(passed=True)

            previous = await uow.benchmark_runs.get_baseline(run)

        if previous is None:
            return RegressionCheckOutput(passed=True, run_id=run.id)

        current = {"metrics": normalize_metrics(run.summary_metrics)}
        baseline = {"metrics": normalize_metrics(previous.summary_metrics)}

        result = history_port.compare_runs(current, baseline)
        return RegressionCheckOutput(
            passed=result["passed"],
            run_id=run.id,
            baseline_run_id=previous.id,
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


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)

    for name in ("faithfulness", "relevancy", "correctness"):
        if name not in result:
            result[name] = result.get(f"avg_{name}")

    return result
