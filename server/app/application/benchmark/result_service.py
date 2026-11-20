"""Application service for benchmark result listing and detail retrieval.

Reads from BenchmarkRunRepository (DB) instead of JSON files on disk.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from operator import eq, ge, lt
from typing import TYPE_CHECKING, Any

from application.benchmark.diagnosis import rediagnose
from application.benchmark.dto import (
    BenchmarkResultDetail,
    BenchmarkResultsList,
    BenchmarkResultSummary,
    BenchmarkReviewPage,
    BenchmarkReviewQuestion,
    RegressionCheckOutput,
    RegressionCheckResult,
)
from application.benchmark.quality import run_replayed_llm
from application.benchmark.regression import regression_report
from domain.benchmark.diagnosis import diagnose, metric, result_key, with_diagnosis
from domain.benchmark.policy import GateStatus


if TYPE_CHECKING:
    from application.benchmark.dto import BenchmarkReviewFilter
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

        run = rediagnose(run)
        summary = result_summary(run)
        return BenchmarkResultDetail(
            id=summary.id,
            summary=summary,
            per_question_results=run.per_question_results,
        )

    async def get_review_summary(self, run_id: int) -> BenchmarkResultSummary | None:
        async with self._uow_factory.create() as uow:
            run = await uow.benchmark_runs.get_review_run(run_id)

        return result_summary(rediagnose(run)) if run is not None else None

    async def list_review_questions(
        self, run_id: int, filters: BenchmarkReviewFilter
    ) -> BenchmarkReviewPage | None:
        async with self._uow_factory.create() as uow:
            run = await uow.benchmark_runs.get_review_run(run_id)

        if run is None:
            return None

        rows = run.per_question_results or []
        query = filters.query.strip().casefold()
        matched: list[BenchmarkReviewQuestion] = []
        scored = 0

        for position, row in enumerate(rows, 1):
            score = review_score(row, filters.metric)
            scored += score is not None
            title = row.get("question") or ""

            if query and query not in title.casefold() and query != str(position):
                continue

            if not matches_review_score(score, filters):
                continue

            diagnosis = diagnose(row).value

            if filters.diagnosis is not None and diagnosis != filters.diagnosis:
                continue

            matched.append(
                BenchmarkReviewQuestion(
                    position=position,
                    key=result_key(row),
                    question=title,
                    diagnosis=diagnosis,
                    correctness=review_score(row, "correctness"),
                    faithfulness=review_score(row, "faithfulness"),
                    hit_rate=review_score(row, "hit_rate"),
                    evidence_metrics=row.get("evidence_metrics") or {},
                )
            )

        if filters.sort != "position":
            direction = -1 if filters.sort == "score_desc" else 1

            def score_order(question: BenchmarkReviewQuestion) -> tuple[bool, float]:
                value = question.correctness if filters.metric == "correctness" else question.faithfulness
                return value is None, direction * (value or 0)

            matched.sort(key=score_order)

        return BenchmarkReviewPage(
            questions=matched[filters.offset : filters.offset + filters.limit],
            total=len(rows),
            matched=len(matched),
            scored=scored,
        )

    async def get_review_question(self, run_id: int, position: int) -> dict[str, Any] | None:
        async with self._uow_factory.create() as uow:
            row = await uow.benchmark_runs.get_result_question(run_id, position - 1)

        if row is None:
            return None

        return {
            **with_diagnosis([row])[0],
            "question_key": result_key(row),
            "correctness": review_score(row, "correctness"),
            "faithfulness": review_score(row, "faithfulness"),
        }

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


def result_summary(run: BenchmarkRun) -> BenchmarkResultSummary:
    if run.id is None:
        message = "BenchmarkRun id is None"
        raise RuntimeError(message)

    return BenchmarkResultSummary(
        id=run.id,
        config_json=run.config_json,
        artifacts=run.artifacts,
        summary_metrics=run.summary_metrics,
        duration_sec=run.duration_sec,
        llm_evaluated=run.llm_evaluated,
        dataset=run.dataset,
        sweep_id=run.sweep_id,
        creation_date=run.creation_date,
    )


def review_score(row: dict[str, Any], name: str) -> float | None:
    group = "retriever_metrics" if name == "hit_rate" else "generator_metrics"
    nested = metric(row, group, name)
    value = nested if nested is not None else row.get(name)

    if isinstance(value, int | float) and not isinstance(value, bool) and isfinite(value):
        return float(value)

    return None


def matches_review_score(value: float | None, filters: BenchmarkReviewFilter) -> bool:
    if filters.comparison == "unscored":
        return value is None

    if filters.score is None:
        return True

    operators = {"eq": eq, "lt": lt, "gte": ge}
    return value is not None and operators[filters.comparison](value, filters.score)
