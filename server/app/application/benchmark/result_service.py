"""Application service for benchmark result listing and detail retrieval.

Reads from BenchmarkRunRepository (DB) instead of JSON files on disk.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from application.benchmark.diagnosis import rediagnose
from application.benchmark.dto import (
    BenchmarkResultDetail,
    BenchmarkResultsList,
    BenchmarkResultSummary,
    RegressionCheckOutput,
    RegressionCheckResult,
)
from application.benchmark.quality import run_replayed_llm
from application.benchmark.regression import regression_report
from domain.benchmark.policy import GateStatus


if TYPE_CHECKING:
    from application.benchmark.ports import BenchmarkHistoryPort
    from application.uow import UnitOfWorkFactory
    from domain.benchmark.run import BenchmarkRun


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

        run = rediagnose(run)

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
        *,
        dataset: str | None = None,
    ) -> RegressionCheckOutput:
        """Compare a DB run (or the dataset's latest) with its baseline and the dataset policy."""
        async with self._uow_factory.create() as uow:
            run = (
                await uow.benchmark_runs.get_by_id(run_id)
                if run_id is not None
                else await uow.benchmark_runs.get_latest(dataset)
            )

            if run is None:
                return RegressionCheckOutput(
                    passed=False,
                    status=GateStatus.INSUFFICIENT_DATA,
                    dataset=dataset,
                    warnings=["no benchmark run to check"],
                )

            previous = await uow.benchmark_runs.get_baseline(run)
            policy = await uow.benchmark_quality.get_policy(run.dataset)
            noise_warning = None

            if policy is not None and policy.noise is not None:
                noise_runs = await uow.benchmark_runs.get_by_ids(policy.noise.run_ids)

                # Sigma saved before replayed runs were rejected would understate the noise.
                if replayed := sorted(r.id for r in noise_runs if r.id is not None and run_replayed_llm(r)):
                    policy = replace(policy, noise=None)
                    noise_warning = (
                        f"stored noise used runs {replayed} with recorded LLM responses; "
                        "default thresholds apply until noise is re-measured with --llm-replay off"
                    )

        report = regression_report(run, previous, policy, history_port)

        if noise_warning is not None:
            report.warnings = [
                noise_warning,
                *(w for w in report.warnings if not w.startswith("noise not measured")),
            ]

        return RegressionCheckOutput(
            passed=report.passed,
            status=report.status,
            run_id=run.id,
            dataset=run.dataset,
            baseline_run_id=previous.id if previous is not None and report.baseline_comparable else None,
            results=[
                RegressionCheckResult(
                    metric=r["metric"],
                    baseline=r.get("baseline"),
                    current=r.get("current"),
                    delta=r.get("delta"),
                    threshold=r["threshold"],
                    failed=r["failed"],
                    note=r.get("note"),
                    kind=r.get("kind", "baseline"),
                )
                for r in report.checks
            ],
            compared_questions=report.compared_questions,
            excluded_questions=report.excluded_questions,
            warnings=report.warnings,
        )
