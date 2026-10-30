"""Offline stage summaries and paired, case-level comparisons."""

from __future__ import annotations

from collections import defaultdict
import json
import random
from statistics import mean
from typing import Any

from domain.benchmark.evaluation import STAGE_NAMES, evaluate_stage_evidence
from domain.benchmark.stats import percentile


def case_values(case: dict[str, Any]) -> dict[str, float | None]:
    evaluation = case.get("stage_evaluation") or evaluate_stage_evidence(
        case.get("annotations"), case.get("evidence"), case.get("answer") or ""
    )
    attempts = evaluation["attempts"]
    stages = attempts[-1]["stages"] if attempts else {}
    values = {f"{name}_evidence_coverage": stages.get(name, {}).get("coverage") for name in STAGE_NAMES}
    values.update(
        {
            f"{name}_all_evidence_present": stages.get(name, {}).get("all_evidence_present")
            for name in STAGE_NAMES
        }
    )

    for kind in ("fact", "condition"):
        values[f"prompt_lexical_{kind}_coverage"] = stages.get("prompt", {}).get(f"lexical_{kind}_coverage")

    for key in (
        "answer_fact_coverage",
        "answer_condition_coverage",
        "refusal_score",
        "citation_support_score",
        "requirement_preservation_score",
    ):
        values[key] = case.get("evidence_metrics", {}).get(key)

    for key in ("faithfulness", "relevancy", "correctness"):
        values[key] = case.get("generator_metrics", {}).get(key)

    for key in ("rag_latency_sec", "rag_cost_usd", "total_cost_usd"):
        values[key] = case.get(key)

    if (case.get("annotation_validation") or {}).get("status") == "invalid_annotation":
        values = {
            key: value if key in {"rag_latency_sec", "rag_cost_usd", "total_cost_usd"} else None
            for key, value in values.items()
        }

    return values


def summarize_stages(cases: list[dict[str, Any]]) -> dict[str, Any]:
    values = [case_values(case) for case in cases]
    keys = sorted({key for row in values for key in row})
    metrics = {}

    for key in keys:
        available = [value for row in values if (value := row[key]) is not None]
        metrics[key] = {"mean": mean(available) if available else None, "valid_count": len(available)}

    return {
        "case_count": len(cases),
        "invalid_annotation_count": sum(
            (case.get("annotation_validation") or {}).get("status") == "invalid_annotation" for case in cases
        ),
        "coverage_method": "lexical; semantic judge scores are separate",
        "metrics": metrics,
    }


def summarize_stage_tags(cases: list[dict[str, Any]]) -> dict[str, Any]:
    tags = sorted({tag for case in cases for tag in case.get("tags") or []})
    return {
        "overall": summarize_stages(cases),
        "by_tags": {
            tag: summarize_stages([case for case in cases if tag in (case.get("tags") or [])]) for tag in tags
        },
    }


def pairing_identity(case: dict[str, Any]) -> str:
    request = (case.get("evidence") or {}).get("request") or {}
    return json.dumps(
        {
            "question": case.get("question"),
            "annotations": case.get("annotations"),
            "history": request.get("history"),
            "as_of_date": request.get("as_of_date"),
            "user": request.get("user"),
            "summary": request.get("summary"),
        },
        sort_keys=True,
        default=str,
    )


def compare_stage_runs(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], *, samples: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Average repetitions per case, then bootstrap paired case differences."""
    if samples < 1:
        message = "bootstrap samples must be positive"
        raise ValueError(message)

    grouped: list[dict[tuple[Any, ...], list[dict[str, Any]]]] = [defaultdict(list), defaultdict(list)]

    for groups, cases in zip(grouped, (baseline, candidate), strict=True):
        for case in cases:
            groups[(case.get("dataset"), str(case["id"]))].append(case)

    left, right = grouped
    pairs = []
    excluded = []

    for key in sorted(left.keys() & right.keys(), key=str):
        identities = {pairing_identity(row) for row in [*left[key], *right[key]]}

        if len(identities) != 1:
            excluded.append({"case": key, "reason": "question_annotation_or_request_scope_changed"})
            continue

        averages = []

        for rows in (left[key], right[key]):
            observations = [case_values(row) for row in rows]
            averaged = {}

            for metric in observations[0]:
                available = [value for obs in observations if (value := obs[metric]) is not None]
                averaged[metric] = mean(available) if available else None

            averages.append(averaged)

        pairs.append((key, averages))

    metrics = paired_metric_differences(pairs, samples, seed)
    return {
        "metrics": metrics,
        "excluded": excluded,
        "paired_cases": len(pairs),
        "baseline_only": sorted(left.keys() - right.keys(), key=str),
        "candidate_only": sorted(right.keys() - left.keys(), key=str),
        "interval_method": "paired case bootstrap; repetitions averaged within case",
        "limitations": [
            "pilot/development cases do not establish held-out improvement",
            "intervals are descriptive and do not correct for multiple comparisons",
        ],
    }


def paired_metric_differences(pairs: list[Any], samples: int, seed: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    rng = random.Random(seed)  # noqa: S311 — reproducible statistical resampling, not security.

    for metric in sorted({key for _, observations in pairs for key in observations[0]}):
        deltas = [
            (key, observations[1][metric] - observations[0][metric])
            for key, observations in pairs
            if observations[0].get(metric) is not None and observations[1].get(metric) is not None
        ]
        values = [value for _, value in deltas]
        boot = (
            sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
            if len(values) > 1
            else []
        )
        lower_is_better = metric.endswith(("latency_sec", "cost_usd"))
        metrics[metric] = {
            "paired_case_count": len(values),
            "mean_delta": mean(values) if values else None,
            "ci95": [percentile(boot, 2.5), percentile(boot, 97.5)] if boot else None,
            "improved": [key for key, delta in deltas if delta < 0]
            if lower_is_better
            else [key for key, delta in deltas if delta > 0],
            "worsened": [key for key, delta in deltas if delta > 0]
            if lower_is_better
            else [key for key, delta in deltas if delta < 0],
        }

    return metrics


def diagnostic_card(case: dict[str, Any]) -> str:
    evaluation = case["stage_evaluation"]
    lines = [
        f"Кейс {case.get('id')}: {case.get('question', '')}",
        f"Разметка: {evaluation['annotation_status']}",
    ]

    for attempt in evaluation["attempts"]:
        lines.append(f"Попытка {attempt['attempt']}:")

        for name, stage in attempt["stages"].items():
            coverage = (
                f"{len(stage['found'])}/{stage['required_count']}"
                if stage.get("found") is not None
                else "нет разметки"
                if stage["status"] == "executed"
                else stage["status"]
            )
            lines.append(f"  {name}: {coverage}")

    for kind, metric in evaluation.get("answer_requirements", {}).items():
        coverage = (
            f"{metric['found_count']}/{metric['required_count']}"
            if metric["required_count"]
            else "нет разметки"
        )
        lines.append(f"В ответе ({kind}, лексически): {coverage}")

    for issue in evaluation["issues"]:
        if issue["kind"] == "lexical_answer_omission":
            lines.append(
                f"В prompt присутствует, в ответе лексически отсутствует: {issue['requirement']}. "
                "Нужна смысловая проверка."
            )
            continue

        label = evaluation["evidence_labels"][issue["fragment"]]
        reference = label.get("alternatives", [label])[0]
        prefix = f"{reference['source']}: {reference['text']}"

        if issue["kind"] == "not_found_before_reranking":
            lines.append(f"Не найдено до reranking (попытка {issue['attempt']}): {prefix}")
        else:
            decision = ", ".join(sorted({item["reason"] for item in issue.get("decisions", [])}))
            action = "Восстановлено" if issue["kind"] == "evidence_recovered" else "Потеряно"
            lines.append(
                f"{action} {issue['from']} → {issue['to']}: {prefix}; "
                f"{decision or 'причина не зафиксирована'}"
            )

    return "\n".join(lines)
