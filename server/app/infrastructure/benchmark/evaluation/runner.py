"""Bind benchmark provider adapters to the Application execution workflow."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Unpack

from application.benchmark.execution import (
    BenchmarkExecutionScope,
    execute_benchmark,
    validate_questions,
)
from application.benchmark.identity import fingerprint
from config import BenchmarkSection, process_section
from infrastructure.benchmark.corpus import benchmark_corpus, corpus_metadata
from infrastructure.benchmark.evaluation import answer_generators
from infrastructure.benchmark.evaluation.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.evaluation.run_manifest import build_run_manifest
from infrastructure.benchmark.judging.judge_rubric import JUDGE_RUBRIC_VERSION
from infrastructure.benchmark.reporting import progress
from infrastructure.benchmark.sweep.settings import settings_fingerprint


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from application.benchmark.dto import BenchmarkRunnerOptions
    from application.ports.ml_clients import MLClientPort
    from infrastructure.benchmark.evaluation.annotation_validation import AnnotationValidator
    from infrastructure.budget.runtime import BudgetRuntime


async def run_benchmark_async(
    questions: list[dict[str, Any]],
    top_k: int,
    judge_model: str,
    *,
    corpus_resource: MLClientPort | None = None,
    budget_runtime: BudgetRuntime | None = None,
    annotation_validator: AnnotationValidator | None = None,
    **options: Unpack[BenchmarkRunnerOptions],
) -> list[dict[str, Any]]:
    rag_service = options.get("rag_service")
    max_concurrent = options.get("max_concurrent")
    seed = options.get("seed")
    n_runs = options.get("n_runs", 1)
    fetch_k = options.get("fetch_k")
    resume = options.get("resume", False)
    checkpoints = options.get("checkpoints")
    checkpoint_prefix = options.get("checkpoint_prefix")

    if rag_service is None:
        message = "rag_service is required for benchmark evaluation"
        raise ValueError(message)

    @asynccontextmanager
    async def scope() -> AsyncIterator[BenchmarkExecutionScope]:
        async with benchmark_corpus(corpus_resource) as corpus:
            manifest = await asyncio.to_thread(
                build_run_manifest, questions, corpus_metadata(corpus), judge_model
            )
            manifest["execution"] = {
                "top_k": top_k,
                "fetch_k": fetch_k,
                "n_runs": n_runs,
                "seed_requested": seed,
            }
            manifest["configuration_version"] = fingerprint(
                {"settings": settings_fingerprint(), "execution": manifest["execution"]}
            )
            yield BenchmarkExecutionScope(
                BenchmarkCaseEvaluator(
                    answer_generators.RagBenchmarkGenerator(rag_service, top_k, fetch_k),
                    judge_model,
                    checkpoints=checkpoints,
                    budget_runtime=budget_runtime,
                    annotation_validator=annotation_validator,
                    run_manifest=manifest,
                ),
                corpus_metadata(corpus),
            )

    return await execute_benchmark(
        questions,
        top_k,
        judge_model,
        max_concurrent=process_section(BenchmarkSection).benchmark_max_concurrent
        if max_concurrent is None
        else max_concurrent,
        seed=seed,
        n_runs=n_runs,
        fetch_k=fetch_k,
        resume=resume,
        checkpoints=checkpoints,
        checkpoint_prefix=checkpoint_prefix,
        scope=scope,
        reporter=progress,
        settings_identity=fingerprint({"settings": settings_fingerprint(), "evidence_schema": 2}),
        rubric_version=JUDGE_RUBRIC_VERSION,
    )


__all__ = ["validate_questions"]
