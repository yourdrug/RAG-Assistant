"""Bind benchmark provider adapters to the Application execution workflow."""

from __future__ import annotations


from typing import Any, TYPE_CHECKING

from contextlib import asynccontextmanager

from application.benchmark.execution import (
    BenchmarkExecutionScope,
    execute_benchmark,
    validate_questions as validate_questions,
)
from config import settings
from infrastructure.benchmark.corpus import benchmark_corpus, corpus_metadata
from infrastructure.benchmark.evaluation import answer_generators
from infrastructure.benchmark.evaluation.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.judging.judge_rubric import JUDGE_RUBRIC_VERSION
from infrastructure.benchmark.reporting import progress
from infrastructure.benchmark.sweep.settings import settings_fingerprint

if TYPE_CHECKING:
    from application.benchmark import BenchmarkCheckpoints
    from infrastructure.budget.runtime import BudgetRuntime
    from application.ports.ml_clients import MLClientPort
    from application.chat import ChatRAGPort
    from collections.abc import AsyncIterator


async def run_benchmark_async(
    questions: list[dict[str, Any]],
    top_k: int,
    judge_model: str,
    max_concurrent: int | None = None,
    seed: int | None = None,
    n_runs: int = 1,
    rag_service: ChatRAGPort | None = None,
    corpus_resource: MLClientPort | None = None,
    budget_runtime: BudgetRuntime | None = None,
    fetch_k: int | None = None,
    resume: bool = False,
    checkpoints: BenchmarkCheckpoints | None = None,
    checkpoint_prefix: str | None = None,
) -> list[dict[str, Any]]:
    if rag_service is None:
        raise ValueError("rag_service is required for benchmark evaluation")

    @asynccontextmanager
    async def scope() -> AsyncIterator[BenchmarkExecutionScope]:
        async with benchmark_corpus(corpus_resource) as corpus:
            yield BenchmarkExecutionScope(
                BenchmarkCaseEvaluator(
                    answer_generators.RagBenchmarkGenerator(rag_service, top_k, fetch_k),
                    judge_model,
                    checkpoints=checkpoints,
                    budget_runtime=budget_runtime,
                ),
                corpus_metadata(corpus),
            )

    return await execute_benchmark(
        questions,
        top_k,
        judge_model,
        max_concurrent=settings.benchmark_max_concurrent if max_concurrent is None else max_concurrent,
        seed=seed,
        n_runs=n_runs,
        fetch_k=fetch_k,
        resume=resume,
        checkpoints=checkpoints,
        checkpoint_prefix=checkpoint_prefix,
        scope=scope,
        reporter=progress,
        settings_identity=settings_fingerprint(),
        rubric_version=JUDGE_RUBRIC_VERSION,
    )
