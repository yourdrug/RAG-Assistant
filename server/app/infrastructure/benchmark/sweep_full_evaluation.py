"""Run full RAG and judge evaluation for the best retrieval configurations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import math

from domain.services.benchmark_selection import select_fast_candidates
from domain.value_objects.benchmark_scoring import METRIC_SCALES, validate_objective_weights
from domain.value_objects.sweep_evaluation_mode import SweepEvaluationMode

from infrastructure.benchmark.sweep_scoring import compute_composite_score
from infrastructure.benchmark.sweep_settings import SweepSettingsPort
from infrastructure.benchmark.sweep_strategies import (
    ProgressCallback,
    ShouldCancel,
    check_cancelled,
    report_progress,
)


class SweepFullEvaluator:
    def __init__(self, runtime: SweepSettingsPort) -> None:
        self._runtime = runtime

    async def evaluate(
        self,
        results: list[dict],
        top_n_llm: int,
        judge_model: str,
        questions: list[dict],
        weights: dict,
        should_cancel: ShouldCancel | None,
        run_benchmark: Callable[[list[dict], str], Awaitable[dict]],
        *,
        evaluation_mode: SweepEvaluationMode = SweepEvaluationMode.FAST,
        progress_callback: ProgressCallback | None = None,
    ) -> list[dict]:
        weights = validate_objective_weights(weights, require_retrieval=False)
        evaluation_mode = SweepEvaluationMode(evaluation_mode)
        candidates = (
            list(results)
            if evaluation_mode == SweepEvaluationMode.FULL
            else select_fast_candidates(results, top_n_llm)
        )
        await report_progress(progress_callback, 0, len(candidates), {"phase": "full_evaluation"})
        finalists = []
        for result in results:
            result["retrieval_score"] = result.get("composite_score")
            result["composite_score"] = None
        for idx, result in enumerate(candidates, 1):
            await check_cancelled(should_cancel, f"phase B config {idx}/{len(candidates)}")
            with self._runtime.override(result["config"]):
                full = await run_benchmark(questions, judge_model)
            result["full_metrics"] = full
            result["llm_evaluated"] = True
            metrics = {
                key: full.get("hit_rate" if key == "hit_rate" else f"avg_{key}") for key in METRIC_SCALES
            }
            result["evaluation_complete"] = has_complete_objectives(full, metrics, weights)
            if result["evaluation_complete"]:
                result["composite_score"] = compute_composite_score(metrics, weights)
                finalists.append(result)
            await report_progress(
                progress_callback, idx, len(candidates), {**result, "phase": "full_evaluation"}
            )
        await check_cancelled(should_cancel, "after phase B evaluation")
        finalists.sort(key=lambda r: r["composite_score"], reverse=True)
        for result in results:
            result["evaluation_mode"] = evaluation_mode.value
            result["evaluated_config_count"] = len(candidates)
            result["scored_config_count"] = len(finalists)
            result["search_config_count"] = len(results)
        # Keep diagnostics for other configs, but never mix phase A and B scores.
        return finalists + [result for result in results if result["composite_score"] is None]


def has_complete_objectives(full: dict, metrics: dict, weights: dict) -> bool:
    for key, weight in weights.items():
        if weight <= 0:
            continue
        value = metrics[key]
        if value is None or not math.isfinite(float(value)):
            return False
        # Unavailable evidence must not improve an average by removing misses.
        if full.get(f"{key}_evaluated_count", 0) < full.get(f"{key}_expected_count", 0):
            return False
    return True
