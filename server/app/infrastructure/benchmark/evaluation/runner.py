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
from infrastructure.benchmark.space_scope import space_corpus_metadata
from infrastructure.benchmark.sweep.settings import settings_fingerprint


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from application.benchmark.dto import BenchmarkRunnerOptions
    from application.ports.ml_clients import MLClientPort
    from domain.knowledge.space import SpaceRef
    from infrastructure.benchmark.evaluation.annotation_validation import AnnotationValidator


async def run_benchmark_async(
    questions: list[dict[str, Any]],
    top_k: int,
    judge_model: str,
    *,
    corpus_resource: MLClientPort | None = None,
    annotation_validator: AnnotationValidator | None = None,
    space_resolver: Callable[[int | None], Awaitable[SpaceRef]] | None = None,
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
    depth = options.get("depth")

    if options.get("space_id") is not None and space_resolver is None:
        message = "An explicit benchmark space requires a space resolver"
        raise ValueError(message)

    if rag_service is None:
        message = "rag_service is required for benchmark evaluation"
        raise ValueError(message)

    @asynccontextmanager
    async def scope() -> AsyncIterator[BenchmarkExecutionScope]:
        space = await space_resolver(options.get("space_id")) if space_resolver is not None else None

        async with benchmark_corpus(corpus_resource) as corpus:
            metadata = space_corpus_metadata(corpus_metadata(corpus), space)
            manifest = await asyncio.to_thread(build_run_manifest, questions, metadata, judge_model)
            manifest["execution"] = {
                "space_id": space.id if space is not None else None,
                "pack_key": space.pack_key if space is not None else None,
                "top_k": top_k,
                "fetch_k": fetch_k,
                "depth": depth,
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
                    annotation_validator=annotation_validator,
                    run_manifest=manifest,
                ),
                metadata,
                space,
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
        depth=depth,
        scope=scope,
        reporter=progress,
        settings_identity=fingerprint({"settings": settings_fingerprint(), "evidence_schema": 2}),
        rubric_version=JUDGE_RUBRIC_VERSION,
    )


__all__ = ["validate_questions"]
