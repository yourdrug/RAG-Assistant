"""Judge one generated answer, reusing successful checkpoint scores."""

from __future__ import annotations
from typing import TYPE_CHECKING

import asyncio
from infrastructure.benchmark.evaluation.answer_generators import BenchmarkAnswer
from infrastructure.benchmark.judging import judge
from infrastructure.benchmark.judging import grouped_judge
from infrastructure.benchmark.judging import evidence_judge
from config import settings

from infrastructure.benchmark.judging import context_judge
from infrastructure.benchmark.checkpoint import restore_grouped_scores

if TYPE_CHECKING:
    from infrastructure.budget.runtime import BudgetRuntime


async def judge_generated(
    question: dict,
    generated: BenchmarkAnswer,
    annotations: dict,
    stages: dict,
    save_stage,
    *,
    judge_model: str,
    runtime: BudgetRuntime | None = None,
) -> tuple[dict, dict, dict]:

    if settings.benchmark_judge_grouped_enabled:
        saved_scores = restore_grouped_scores(stages)
        grouped = await grouped_judge.judge_case_grouped(
            question,
            generated.answer,
            generated.context,
            evidence_context=generated.evidence.context if generated.evidence else None,
            model=judge_model,
            runtime=runtime,
            saved=saved_scores,
            callback=lambda value: save_stage("grouped_metrics", value),
        )

        def metric_values(keys: tuple[str, ...]) -> dict:
            values = {}

            for key in keys:
                detail = grouped.get(key, {})
                values[key] = detail.get("score")
                values[f"{key}_reason"] = detail.get("reason", "Оценка недоступна")

                if "error" in detail:
                    values[f"{key}_error"] = detail["error"]

            return values

        generator_metrics = metric_values(("faithfulness", "relevancy", "correctness"))
        context_metrics = metric_values(("context_precision", "context_recall"))
        evidence_keys = ("citation_support_score", "refusal_score", "requirement_preservation_score")
        judge_metrics = {
            "scores": {key: grouped[key].get("score") for key in evidence_keys if key in grouped},
            "details": {
                key: {k: v for k, v in grouped[key].items() if k != "score"}
                for key in evidence_keys
                if key in grouped
            },
        }
    else:
        generator_metrics = await judge.judge_answer_async(
            question=question["question"],
            answer=generated.answer,
            context=generated.context,
            expected_answer=question.get("expected_answer"),
            judge_model=judge_model,
            runtime=runtime,
            saved_metrics=stages.get("generator_metrics"),
            metric_callback=lambda value: save_stage("generator_metrics", value),
        )
        loop = asyncio.get_running_loop()

        def save_sync(name: str, value: dict) -> None:
            asyncio.run_coroutine_threadsafe(save_stage(name, value), loop).result()

        context_metrics = (
            await asyncio.to_thread(
                context_judge.compute_context_precision_recall,
                question["question"],
                question.get("expected_answer") or "",
                context_override=generated.context,
                judge_model=judge_model,
                runtime=runtime,
                saved_metrics=stages.get("context_metrics"),
                metric_callback=lambda value: save_sync("context_metrics", value),
            )
            if generated.context
            else {"context_precision": None, "context_recall": None}
        )

        judge_metrics = await asyncio.to_thread(
            evidence_judge.judge_evidence,
            question["question"],
            generated.answer,
            generated.evidence.context if generated.evidence else None,
            annotations,
            judge_model,
            runtime=runtime,
            saved_metrics=stages.get("evidence_judge"),
            metric_callback=lambda value: save_sync("evidence_judge", value),
        )

    return generator_metrics, context_metrics, judge_metrics
