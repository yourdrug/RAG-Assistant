"""Shared full RAG and judge executor for benchmark jobs and sweep finalists."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Unpack, cast

from infrastructure.benchmark.runner import run_benchmark_async


if TYPE_CHECKING:
    from application.benchmark.dto import BenchmarkRunnerOptions


class BenchmarkExecutor:
    async def run(
        self,
        questions: list[dict[str, Any]],
        top_k: int,
        judge_model: str,
        **options: Unpack[BenchmarkRunnerOptions],
    ) -> list[dict[str, Any]]:
        return cast(
            "list[dict[str, Any]]",
            await run_benchmark_async(questions=questions, top_k=top_k, judge_model=judge_model, **options),
        )
