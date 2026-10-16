"""Benchmark runner — main benchmark orchestration.

Runs the full RAG pipeline via RagService.invoke() with cache disabled,
then evaluates quality via LLM judge.
"""

import asyncio
from contextlib import AsyncExitStack
import logging
import time
from pathlib import Path
from typing import cast

from config import get_setting
from infrastructure.benchmark.checkpoint import fingerprint, FileBenchmarkCheckpoints
from infrastructure.benchmark.corpus import benchmark_corpus, corpus_metadata
from infrastructure.benchmark.judge_rubric import JUDGE_RUBRIC_VERSION
from application.ports.benchmark_checkpoints import BenchmarkCheckpoints

from config import settings
from domain.value_objects.benchmark_annotations import validate_annotations
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.roles import UserKind, UserRole
from infrastructure.benchmark import answer_generators, persistence
from infrastructure.benchmark.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.metrics import result_judge_errors

from infrastructure.benchmark.case_metrics import (
    compute_retriever_metrics_from_sources as compute_retriever_metrics_from_sources,
)
from infrastructure.benchmark.case_metrics import (
    extract_usage_from_response as extract_usage_from_response,
)

logger = logging.getLogger("default")


def validate_questions(questions: list[dict]) -> None:
    """Validate database cases before starting expensive evaluation."""
    if not isinstance(questions, list):
        raise ValueError("benchmark questions must be a list")
    if not questions:
        raise ValueError("benchmark requires at least one active question")
    for question in questions:
        if not isinstance(question, dict) or not isinstance(question.get("question"), str):
            raise ValueError("each benchmark case requires a question string")
        validate_annotations(question.get("annotations"))


def validate_pipeline(rag_service, standalone: bool) -> None:
    if rag_service is None and not standalone:
        raise ValueError("rag_service is required; use standalone=True for standalone evaluation")
    if rag_service is not None and standalone:
        raise ValueError("rag_service and standalone=True are mutually exclusive")


async def close_owned_generator(generator, standalone: bool) -> None:
    if standalone:
        await generator.close()


def validate_output_options(resume, checkpoints, export_files, out_dir) -> None:
    if (resume and checkpoints is None or export_files) and out_dir is None:
        raise ValueError("Local checkpoint/export requires an explicit output directory")


async def prepare_checkpoint_corpus(store, checkpoint_dir: Path | None, corpus) -> Path | None:
    if store is None or checkpoint_dir is None or corpus is None:
        return checkpoint_dir
    corpus_dir = checkpoint_dir / corpus.namespace
    corpus_key = str(corpus_dir / "corpus.json")
    saved_corpus = await store.load(corpus_key)
    metadata = corpus_metadata(corpus)
    if saved_corpus is not None and saved_corpus != metadata:
        raise ValueError("Benchmark corpus changed; start a new run")
    await store.save(corpus_key, metadata)
    return corpus_dir


async def run_benchmark_async(
    questions: list[dict],
    out_dir: str | None,
    top_k: int,
    judge_model: str,
    max_concurrent: int | None = None,
    seed: int | None = None,
    n_runs: int = 1,
    rag_service=None,
    fetch_k: int | None = None,
    resume: bool = False,
    checkpoints: BenchmarkCheckpoints | None = None,
    checkpoint_prefix: str | None = None,
    export_files: bool = False,
    standalone: bool = False,
) -> list[dict]:
    """Schedule benchmark cases, aggregate each run and persist the results."""
    validate_pipeline(rag_service, standalone)
    if max_concurrent is None:
        max_concurrent = settings.benchmark_max_concurrent
    logger.info(
        "RAG Benchmark (async, max_concurrent=%d, pipeline=%s)",
        max_concurrent,
        "RagService" if rag_service else "standalone",
    )
    logger.info(
        "questions=%d top_k=%d model=%s judge=%s seed=%s runs=%d",
        len(questions),
        top_k,
        settings.llm_model,
        judge_model,
        seed,
        n_runs,
    )
    validate_questions(questions)
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    ctx = ChatContext(user_id=0, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN)
    generator = (
        answer_generators.RagBenchmarkGenerator(rag_service, top_k, fetch_k)
        if rag_service is not None
        else answer_generators.StandaloneBenchmarkGenerator(top_k, fetch_k, ctx)
    )
    stack = AsyncExitStack()
    try:
        stack.push_async_callback(close_owned_generator, generator, standalone)
        registry = getattr(generator, '_clients', None) if standalone else getattr(rag_service, '_ml', None)
        corpus = await stack.enter_async_context(benchmark_corpus(registry))
        # Store only a hash of settings: credentials must never enter report files.
        identity = fingerprint(
            {
                "corpus": corpus_metadata(corpus),
                "questions": questions,
                "top_k": top_k,
                "fetch_k": fetch_k,
                "judge": judge_model,
                "seed": seed,
                "runs": n_runs,
                "settings": {
                    name: get_setting(name)
                    for name in type(settings).model_fields
                    if not any(part in name for part in ("key", "password", "secret", "url", "dir"))
                    and name not in {"benchmark_judge_initial_tokens", "benchmark_judge_retry_tokens"}
                },
            }
        )
        checkpoint_dir = Path("checkpoints") / (checkpoint_prefix or identity) if resume else None
        validate_output_options(resume, checkpoints, export_files, out_dir)
        store = checkpoints or (FileBenchmarkCheckpoints(Path(out_dir)) if resume and out_dir else None)
        checkpoint_dir = await prepare_checkpoint_corpus(store, checkpoint_dir, corpus)
        evaluator = BenchmarkCaseEvaluator(generator, judge_model, checkpoints=store)
        all_results = []

        async def process_question(idx: int, question: dict, run_idx: int) -> dict:
            nonlocal completed
            async with semaphore:
                logger.info(
                    "Benchmark question %s started (run %d, slot limit %d)",
                    question.get("id", idx),
                    run_idx,
                    max_concurrent,
                )
                path = checkpoint_dir / f"{run_idx}-{idx}.json" if checkpoint_dir else None
                cached = cast(dict | None, await store.load(str(path))) if path and store else None
                if cached is not None and cached.get("judge_rubric_version") == JUDGE_RUBRIC_VERSION:
                    completed += 1
                    logger.info(
                        "Benchmark progress: %d/%d completed (restored question=%s)",
                        completed,
                        len(questions) * n_runs,
                        question.get("id", idx),
                    )
                    return cached
                result = await evaluator.evaluate(
                    idx,
                    question,
                    run_idx,
                    ctx,
                    **({"checkpoint_path": path.with_suffix(".stages.json")} if path else {}),
                )
                diagnostic_errors = bool(result_judge_errors(result))
                if path and store is not None and not diagnostic_errors:
                    await store.save(str(path), result)
                completed += 1
                logger.info(
                    "Benchmark progress: %d/%d completed; question=%s",
                    completed,
                    len(questions) * n_runs,
                    question.get("id", idx),
                )
                persistence.log_question_result(idx, len(questions), question, result)
                return result

        for run_idx in range(1, n_runs + 1):
            started = time.monotonic()
            # TaskGroup cancels sibling work on failure; gather previously left paid
            # requests running while the worker restarted the entire sweep.
            try:
                async with asyncio.TaskGroup() as group:
                    tasks = [
                        group.create_task(process_question(idx, q, run_idx))
                        for idx, q in enumerate(questions, 1)
                    ]
            except ExceptionGroup as exc:
                # Preserve the actionable provider error in the persisted job error.
                failure = exc.exceptions[0]
                while isinstance(failure, ExceptionGroup):
                    failure = failure.exceptions[0]
                raise failure from exc
            results = [task.result() for task in tasks]
            persistence.log_summary(list(results), time.monotonic() - started)
            all_results.extend(results)
        if export_files:
            await asyncio.to_thread(
                persistence.save_results,
                all_results,
                cast(str, out_dir),
                model_name=settings.llm_model,
                run_id="all" if n_runs > 1 else "",
                corpus=corpus_metadata(corpus),
            )
        return all_results
    finally:
        await stack.aclose()
