"""Benchmark runner — main benchmark orchestration.

Runs the full RAG pipeline via RagService.invoke() with cache disabled,
then evaluates quality via LLM judge.
"""

import asyncio
import logging

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
):
    """Schedule benchmark cases, aggregate each run and persist the results."""
    import time

    from domain.value_objects.chat_context import ChatContext

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
    semaphore = asyncio.Semaphore(max_concurrent)
    ctx = ChatContext(user_id=0, user_kind="internal", user_role="admin")
    generator = (
        RagBenchmarkGenerator(rag_service, top_k, fetch_k)
        if rag_service is not None
        else StandaloneBenchmarkGenerator(top_k, fetch_k, ctx)
    )
    evaluator = BenchmarkCaseEvaluator(generator, judge_model)
    all_results = []

    async def process_question(idx: int, question: dict, run_idx: int) -> dict:
        async with semaphore:
            result = await evaluator.evaluate(idx, question, run_idx, ctx)
            log_question_result(idx, len(questions), question, result)
            return result

    for run_idx in range(1, n_runs + 1):
        started = time.monotonic()
        results = await asyncio.gather(
            *(process_question(idx, q, run_idx) for idx, q in enumerate(questions, 1))
        )
        log_summary(list(results), time.monotonic() - started)
        all_results.extend(results)
    await asyncio.to_thread(
        save_results, all_results, out_dir, model_name=settings.llm_model, run_id="all" if n_runs > 1 else ""
    )
