"""Regression of a run against its baseline on unchanged questions, plus dataset acceptance floors."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any

from application.benchmark.diagnosis import rediagnose
from application.benchmark.summary import summarize_benchmark_results
from domain.benchmark.diagnosis import result_key
from domain.benchmark.policy import (
    AcceptanceThresholds,
    GateStatus,
    acceptance_checks,
    allowed_drops,
    question_weights,
    run_judge,
)


if TYPE_CHECKING:
    from application.benchmark.ports import BenchmarkHistoryPort
    from domain.benchmark.policy import DatasetPolicy
    from domain.benchmark.run import BenchmarkRun


logger = logging.getLogger(__name__)

ANSWER_METRICS = ("faithfulness", "relevancy", "correctness")


@dataclass
class RegressionReport:
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    compared_questions: int | None = None
    excluded_questions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    baseline_comparable: bool = True
    status: GateStatus = GateStatus.PASSED


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)

    for name in ANSWER_METRICS:
        if name not in result:
            result[name] = result.get(f"avg_{name}")

    return result


def comparable_rows(
    current: list[dict[str, Any]] | None, baseline: list[dict[str, Any]] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Rows of cases whose content is unchanged since the baseline, and the excluded cases.

    None when either run predates per-question hashes and cannot be matched safely.
    """
    if not current or not baseline:
        return None

    if any(not row.get("question_hash") for row in (*current, *baseline)):
        return None

    before = {result_key(row): row for row in baseline}
    after = {result_key(row): row for row in current}
    kept_current, kept_baseline, excluded = [], [], []

    for key, row in after.items():
        previous = before.get(key)

        if previous is None:
            excluded.append({"key": key, "question": row.get("question"), "reason": "added"})
        elif previous["question_hash"] != row["question_hash"]:
            excluded.append({"key": key, "question": row.get("question"), "reason": "changed"})
        else:
            kept_current.append(row)
            kept_baseline.append(previous)

    excluded.extend(
        {"key": key, "question": row.get("question"), "reason": "removed"}
        for key, row in before.items()
        if key not in after
    )
    return kept_current, kept_baseline, excluded


def subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        return summarize_benchmark_results(rows)
    except (KeyError, TypeError):
        # Rows of old formats lack fields the current summary needs; compare stored summaries instead.
        logger.warning("Benchmark rows cannot be summarized again; stored summaries are compared")
        return None


def matched_metrics(
    run: BenchmarkRun,
    baseline: BenchmarkRun,
    matched: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
    report: RegressionReport,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    kept_current, kept_baseline, excluded = matched
    report.excluded_questions = excluded
    report.compared_questions = len(kept_current)

    if not excluded:
        return run.summary_metrics, baseline.summary_metrics

    current = subset_summary(kept_current) if kept_current else None
    previous = subset_summary(kept_baseline) if kept_baseline else None

    if current is None or previous is None:
        report.warnings.append("no unchanged questions could be summarized; baseline not compared")
        return None

    return current, previous


def baseline_metrics(
    run: BenchmarkRun, baseline: BenchmarkRun, report: RegressionReport
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    matched = comparable_rows(run.per_question_results, baseline.per_question_results)

    if matched is not None:
        return matched_metrics(run, baseline, matched, report)

    if run.config_json.get("dataset_hash") != baseline.config_json.get("dataset_hash"):
        report.warnings.append("baseline used different questions without per-question hashes; not compared")
        return None

    report.warnings.append("runs predate per-question hashes; whole datasets compared")
    return run.summary_metrics, baseline.summary_metrics


def regression_report(
    run: BenchmarkRun,
    baseline: BenchmarkRun | None,
    policy: DatasetPolicy | None,
    history: BenchmarkHistoryPort,
) -> RegressionReport:
    run = rediagnose(run)
    report = RegressionReport(passed=False)
    thresholds = policy.thresholds if policy is not None else AcceptanceThresholds()
    report.checks.extend(acceptance_checks(normalize_metrics(run.summary_metrics), thresholds))
    metrics = comparable_baseline(run, baseline, report)

    if metrics is None:
        report.baseline_comparable = False
    else:
        judge = run_judge(run.config_json, run.per_question_results)
        drops, warnings = allowed_drops(
            policy.noise if policy is not None else None, judge, question_weights(metrics[0])
        )
        report.warnings.extend(warnings)
        compared = history.compare_runs(
            {"metrics": normalize_metrics(metrics[0])},
            {"metrics": normalize_metrics(metrics[1])},
            drops,
        )
        baseline_checks = [{**check, "kind": "baseline"} for check in compared["results"]]
        report.checks.extend(baseline_checks)

        if all(check.get("delta") is None for check in baseline_checks):
            report.warnings.append("no metric could be compared with the baseline")
            report.baseline_comparable = False

    report.status = gate_status(report)
    report.passed = report.status is GateStatus.PASSED
    return report


def comparable_baseline(
    run: BenchmarkRun, baseline: BenchmarkRun | None, report: RegressionReport
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if baseline is None:
        report.warnings.append("no compatible baseline run")
        return None

    if run.llm_evaluated or baseline.llm_evaluated:
        current_judge = run_judge(run.config_json, run.per_question_results)
        baseline_judge = run_judge(baseline.config_json, baseline.per_question_results)

        # Scores of different judges or rubrics are not on one scale.
        if current_judge != baseline_judge:
            report.warnings.append(
                f"baseline was judged by {baseline_judge.model} (rubric {baseline_judge.rubric_version}), "
                f"this run by {current_judge.model} (rubric {current_judge.rubric_version}); not compared"
            )
            return None

    return baseline_metrics(run, baseline, report)


def gate_status(report: RegressionReport) -> GateStatus:
    """A failed check fails the gate; otherwise any unchecked part leaves it without enough data."""
    if any(check["failed"] for check in report.checks):
        return GateStatus.FAILED

    if not report.baseline_comparable or any(check.get("insufficient") for check in report.checks):
        return GateStatus.INSUFFICIENT_DATA

    return GateStatus.PASSED
