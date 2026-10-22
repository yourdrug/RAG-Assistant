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

from infrastructure.benchmark.checkpoint import fingerprint
from infrastructure.benchmark.corpus import benchmark_corpus, corpus_metadata
from infrastructure.benchmark.judging.judge_rubric import JUDGE_RUBRIC_VERSION
from application.benchmark import BenchmarkCheckpoints

from config import settings
from domain.benchmark.question import validate_annotations
from domain.rag.context import RagContext
from domain.identity.roles import UserKind, UserRole
from infrastructure.benchmark.evaluation import answer_generators
from infrastructure.benchmark.reporting import progress
from infrastructure.benchmark.evaluation.case_evaluator import BenchmarkCaseEvaluator
from domain.benchmark.judge import result_judge_errors

from infrastructure.benchmark.sweep.settings import settings_fingerprint

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
    top_k: int,
    judge_model: str,
    max_concurrent: int | None = None,
    seed: int | None = None,
    n_runs: int = 1,
    rag_service=None,
    corpus_resource=None,
    fetch_k: int | None = None,
    resume: bool = False,
    checkpoints: BenchmarkCheckpoints | None = None,
    checkpoint_prefix: str | None = None,
) -> list[dict]:
    """Schedule benchmark cases, aggregate each run and persist the results."""
    if rag_service is None:
        raise ValueError("rag_service is required for benchmark evaluation")

    if resume and checkpoints is None:
        raise ValueError("A database checkpoint store is required for resume")

    if max_concurrent is None:
        max_concurrent = settings.benchmark_max_concurrent

    logger.info(
        "RAG Benchmark (async, max_concurrent=%d, pipeline=%s)",
        max_concurrent,
        "RagService",
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
    ctx = RagContext(user_id=0, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN)
    generator = answer_generators.RagBenchmarkGenerator(rag_service, top_k, fetch_k)
    stack = AsyncExitStack()
    try:
        corpus = await stack.enter_async_context(benchmark_corpus(corpus_resource))
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
                "settings": settings_fingerprint(),
            }
        )
        checkpoint_dir = Path("checkpoints") / (checkpoint_prefix or identity) if resume else None
        store = checkpoints
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
                progress.log_question_result(idx, len(questions), question, result)
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
            progress.log_summary(list(results), time.monotonic() - started)
            all_results.extend(results)

        return all_results
    finally:
        await stack.aclose()
