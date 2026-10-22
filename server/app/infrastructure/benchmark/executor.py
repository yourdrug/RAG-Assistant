"""Shared full RAG and judge executor for benchmark jobs and sweep finalists."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Unpack, cast

from infrastructure.benchmark.evaluation.runner import run_benchmark_async


if TYPE_CHECKING:
    from application.benchmark.dto import BenchmarkRunnerOptions
    from application.ports.ml_clients import MLClientPort


class BenchmarkExecutor:
    def __init__(self, corpus_resource: MLClientPort | None = None) -> None:
        self.corpus_resource = corpus_resource

    async def run(
        self,
        questions: list[dict[str, Any]],
        top_k: int,
        judge_model: str,
        **options: Unpack[BenchmarkRunnerOptions],
    ) -> list[dict[str, Any]]:
        return cast(
            "list[dict[str, Any]]",
            await run_benchmark_async(
                questions=questions,
                top_k=top_k,
                judge_model=judge_model,
                corpus_resource=self.corpus_resource,
                **options,
            ),
        )
