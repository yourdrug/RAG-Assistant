"""Run full RAG and judge evaluation for the best retrieval configurations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from domain.value_objects.benchmark_scoring import validate_objective_weights

from infrastructure.benchmark.sweep_scoring import compute_composite_score
from infrastructure.benchmark.sweep_settings import SweepSettingsPort
from infrastructure.benchmark.sweep_strategies import ShouldCancel, check_cancelled


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
    ) -> list[dict]:
        weights = validate_objective_weights(weights)
        finalists = []
        for result in results:
            result["retrieval_score"] = result.get("composite_score")
            result["composite_score"] = None
        for idx, result in enumerate(results[:top_n_llm], 1):
            await check_cancelled(should_cancel, f"phase B config {idx}/{top_n_llm}")
            with self._runtime.override(result["config"]):
                full = await run_benchmark(questions, judge_model)
            result["full_metrics"] = full
            result["llm_evaluated"] = True
            metrics = {
                "hit_rate": full.get("hit_rate"),
                "mrr": full.get("avg_mrr"),
                "faithfulness": full.get("avg_faithfulness"),
                "relevancy": full.get("avg_relevancy"),
                "correctness": full.get("avg_correctness"),
            }
            if all(metrics[key] is not None for key, weight in weights.items() if weight > 0):
                result["composite_score"] = compute_composite_score(metrics, weights)
                finalists.append(result)
        if not finalists:
            raise ValueError("No fully evaluated configurations have all positively weighted metrics")
        finalists.sort(key=lambda r: r["composite_score"], reverse=True)
        # Keep diagnostics for other configs, but never mix phase A and B scores.
        return finalists + [result for result in results if result["composite_score"] is None]
