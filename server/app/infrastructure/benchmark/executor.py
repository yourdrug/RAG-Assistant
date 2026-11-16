"""Shared full RAG and judge executor for benchmark jobs and sweep finalists."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Unpack

from infrastructure.benchmark.evaluation.runner import run_benchmark_async


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from application.benchmark.dto import BenchmarkRunnerOptions
    from application.ports.ml_clients import MLClientPort
    from domain.knowledge.space import SpaceRef
    from infrastructure.benchmark.evaluation.annotation_validation import AnnotationValidator


class BenchmarkExecutor:
    def __init__(
        self,
        corpus_resource: MLClientPort | None = None,
        *,
        annotation_validator: AnnotationValidator | None = None,
        space_resolver: Callable[[int | None], Awaitable[SpaceRef]] | None = None,
    ) -> None:
        self.annotation_validator = annotation_validator
        self.corpus_resource = corpus_resource
        # Space the benchmark questions are asked in; resolved when a run starts.
        self.space_resolver = space_resolver

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
            annotation_validator=self.annotation_validator,
            space_resolver=self.space_resolver,
            **options,
        )
