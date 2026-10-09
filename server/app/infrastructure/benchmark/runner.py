"""Benchmark runner — main benchmark orchestration.

Runs the full RAG pipeline via RagService.invoke() with cache disabled,
then evaluates quality via LLM judge.
"""

import asyncio
import logging
from pathlib import Path

from config import get_setting
from infrastructure.benchmark.checkpoint import fingerprint, FileBenchmarkCheckpoints
from infrastructure.benchmark.judge_rubric import JUDGE_RUBRIC_VERSION
from application.ports.benchmark_checkpoints import BenchmarkCheckpoints

from config import settings
from domain.value_objects.benchmark_annotations import validate_annotations

from infrastructure.benchmark.case_metrics import (
    compute_retriever_metrics_from_sources as compute_retriever_metrics_from_sources,
)
from infrastructure.benchmark.case_metrics import (
    _extract_usage_from_response as _extract_usage_from_response,
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


async def run_benchmark_async(
    questions: list[dict],
    out_dir: str,
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
    export_files: bool = True,
) -> list[dict]:
    """Schedule benchmark cases, aggregate each run and persist the results."""
    import time

    from domain.value_objects.chat_context import ChatContext
    from domain.value_objects.roles import UserKind, UserRole

    from infrastructure.benchmark.answer_generators import RagBenchmarkGenerator, StandaloneBenchmarkGenerator
    from infrastructure.benchmark.case_evaluator import BenchmarkCaseEvaluator
    from infrastructure.benchmark.persistence import log_question_result, log_summary, save_results

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
        RagBenchmarkGenerator(rag_service, top_k, fetch_k)
        if rag_service is not None
        else StandaloneBenchmarkGenerator(top_k, fetch_k, ctx)
    )
    # Store only a hash of settings: credentials must never enter report files.
    identity = fingerprint(
        {
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
    store = checkpoints or (FileBenchmarkCheckpoints(Path(out_dir)) if resume else None)
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
            cached = await store.load(str(path)) if path and store else None
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
            from infrastructure.benchmark.metrics import result_judge_errors

            diagnostic_errors = bool(result_judge_errors(result))
            if path and not diagnostic_errors:
                await store.save(str(path), result)
            completed += 1
            logger.info(
                "Benchmark progress: %d/%d completed; question=%s",
                completed,
                len(questions) * n_runs,
                question.get("id", idx),
            )
            log_question_result(idx, len(questions), question, result)
            return result

    for run_idx in range(1, n_runs + 1):
        started = time.monotonic()
        # TaskGroup cancels sibling work on failure; gather previously left paid
        # requests running while the worker restarted the entire sweep.
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(process_question(idx, q, run_idx)) for idx, q in enumerate(questions, 1)
                ]
        except ExceptionGroup as exc:
            # Preserve the actionable provider error in the persisted job error.
            failure = exc.exceptions[0]
            while isinstance(failure, ExceptionGroup):
                failure = failure.exceptions[0]
            raise failure from exc
        results = [task.result() for task in tasks]
        log_summary(list(results), time.monotonic() - started)
        all_results.extend(results)
    if export_files:
        await asyncio.to_thread(
            save_results,
            all_results,
            out_dir,
            model_name=settings.llm_model,
            run_id="all" if n_runs > 1 else "",
        )
    return all_results
