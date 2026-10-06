"""Run full RAG and judge evaluation for the best retrieval configurations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

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
        for idx, result in enumerate(results[:top_n_llm], 1):
            await check_cancelled(should_cancel, f"phase B config {idx}/{top_n_llm}")
            with self._runtime.override(result["config"]):
                full = await run_benchmark(questions, judge_model)
            result["full_metrics"] = full
            result["llm_evaluated"] = True
            result["composite_score"] = compute_composite_score(
                {
                    "hit_rate": result.get("avg_hit_rate", 0),
                    "mrr": result.get("avg_mrr", 0),
                    "faithfulness": full.get("avg_faithfulness", 0),
                    "relevancy": full.get("avg_relevancy", 0),
                    "correctness": full.get("avg_correctness"),
                },
                weights,
            )
        results.sort(key=lambda r: r.get("composite_score", 0), reverse=True)
        return results
