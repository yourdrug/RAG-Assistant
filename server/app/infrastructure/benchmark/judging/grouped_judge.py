"""Three independent judge requests with per-metric validation and recovery."""

from infrastructure.benchmark.judging.clients import create_async_judge_client
from infrastructure.benchmark.judging.requests import judge_group

from infrastructure.budget.runtime import BudgetRuntime
import asyncio
import time



from infrastructure.benchmark.judging.judge_diagnostics import (
    log_judge_event,
)



# Semaphores belong to an event loop and cap judge requests across all cases.












async def judge_case_grouped(
    question: dict,
    answer: str,
    context: str,
    *,
    evidence_context: str | None,
    model: str,
    runtime: BudgetRuntime | None = None,
    saved: dict | None = None,
    callback=None,
) -> dict:
    started = time.monotonic()
    annotations = question.get("annotations") or {}
    scores = dict(saved or {})

    async def persist(value: dict) -> None:
        if callback is not None:
            await callback(value)

    quality = ["relevancy"]

    if question.get("expected_answer"):
        quality.append("correctness")

    if "expected_refusal" in annotations:
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
            },
            quality,
        ),
        (
            {
                "question": question["question"],
                "answer": answer,
                "context": context,
                "required_facts": annotations.get("required_facts"),
                "required_conditions": annotations.get("required_conditions"),
            },
            support,
        ),
    ]

    if context:
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

    async with create_async_judge_client(runtime=runtime) as client:
        async with asyncio.TaskGroup() as tasks:
            for payload, metrics in groups:
                tasks.create_task(judge_group(client, model, payload, metrics, scores, persist))
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
