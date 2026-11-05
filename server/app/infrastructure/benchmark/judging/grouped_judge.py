"""Three independent judge requests with per-metric validation and recovery."""

import asyncio
import time
from typing import Any, Unpack

from domain.benchmark.question import ABSTENTION_TYPES, ANSWER_BEHAVIORS, answer_type
from infrastructure.benchmark.judging.callbacks import GroupedMetricOptions
from infrastructure.benchmark.judging.clients import create_async_judge_client
from infrastructure.benchmark.judging.judge_diagnostics import (
    log_judge_event,
)
from infrastructure.benchmark.judging.judge_rubric import ANSWER_TYPE_RUBRICS
from infrastructure.benchmark.judging.requests import judge_group
from infrastructure.budget.runtime import BudgetRuntime


# Semaphores belong to an event loop and cap judge requests across all cases.


async def judge_case_grouped(
    question: dict[str, Any],
    answer: str,
    context: str,
    *,
    evidence_context: str | None,
    model: str,
    runtime: BudgetRuntime | None = None,
    **options: Unpack[GroupedMetricOptions],
) -> dict[str, Any]:
    started = time.monotonic()
    scores = dict(options.get("saved") or {})
    callback = options.get("callback")

    async def persist(value: dict[str, Any]) -> None:
        if callback is not None:
            await callback(value)

    groups = judge_groups(question, answer, context, evidence_context)

    async with create_async_judge_client(runtime=runtime) as client, asyncio.TaskGroup() as tasks:
        for payload, metrics in groups:
            _ = tasks.create_task(
                judge_group(client, model, payload, metrics, saved=scores, callback=persist)
            )
    missing = [key for key, value in scores.items() if value.get("score") is None]
    log_judge_event(
        {
            "event": "judge_case_completed",
            **({"question_id": question["id"]} if "id" in question else {}),
            "model": model,
            "outcome": "partial" if missing else "success",
            "missing_metrics": missing,
            "scored_metric_count": len(scores) - len(missing),
            "elapsed_sec": round(time.monotonic() - started, 3),
        },
        warning=bool(missing),
    )
    return scores


def judge_groups(
    question: dict[str, Any], answer: str, context: str, evidence_context: str | None
) -> list[tuple[dict[str, Any], list[str]]]:
    annotations = question.get("annotations") or {}
    kind = answer_type(annotations)
    rubric = {
        "answer_type": kind,
        "expected_behavior": ANSWER_BEHAVIORS[kind],
        "answer_rubric": ANSWER_TYPE_RUBRICS[kind],
    }
    quality = ["relevancy"]

    if question.get("expected_answer"):
        quality.append("correctness")

    if "expected_refusal" in annotations or "answer_type" in annotations:
        quality.append("refusal_score")

    support = ["faithfulness"]

    if annotations and evidence_context is not None:
        support.append("citation_support_score")

    if annotations.get("required_facts") or annotations.get("required_conditions"):
        support.append("requirement_preservation_score")

    groups = [
        (
            {
                "question": question["question"],
                "answer": answer,
                "expected_answer": question.get("expected_answer"),
                "expected_refusal": annotations.get("expected_refusal"),
                **rubric,
                "reference_evidence": annotations.get("gold_evidence")
                or annotations.get("expected_fragments"),
            },
            quality,
        ),
        (
            {
                "question": question["question"],
                "answer": answer,
                "context": context,
                **rubric,
                "required_facts": annotations.get("required_facts"),
                "required_conditions": annotations.get("required_conditions"),
            },
            support,
        ),
    ]

    if context and kind not in ABSTENTION_TYPES:
        groups.append(
            (
                {
                    "question": question["question"],
                    "context": context,
                    "expected_answer": question.get("expected_answer"),
                },
                ["context_precision", "context_recall"],
            )
        )

    return groups
