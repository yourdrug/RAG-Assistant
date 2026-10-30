"""Shared full RAG and judge executor for benchmark jobs and sweep finalists."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Unpack

from infrastructure.benchmark.evaluation.runner import run_benchmark_async


if TYPE_CHECKING:
    from application.benchmark.dto import BenchmarkRunnerOptions
    from application.ports.ml_clients import MLClientPort
    from infrastructure.benchmark.evaluation.annotation_validation import AnnotationValidator
    from infrastructure.budget.runtime import BudgetRuntime


class BenchmarkExecutor:
    def __init__(
        self,
        corpus_resource: MLClientPort | None = None,
        *,
        budget_runtime: BudgetRuntime | None = None,
        annotation_validator: AnnotationValidator | None = None,
    ) -> None:
        self.annotation_validator = annotation_validator
        self.budget_runtime = budget_runtime
        self.corpus_resource = corpus_resource

    async def run(
        self,
        questions: list[dict[str, Any]],
        top_k: int,
        judge_model: str,
        **options: Unpack[BenchmarkRunnerOptions],
    ) -> list[dict[str, Any]]:
        return await run_benchmark_async(
            questions=questions,
            top_k=top_k,
            judge_model=judge_model,
            corpus_resource=self.corpus_resource,
            budget_runtime=self.budget_runtime,
            annotation_validator=self.annotation_validator,
            **options,
        )
