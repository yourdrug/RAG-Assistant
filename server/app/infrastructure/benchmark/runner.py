"""Benchmark runner — main benchmark orchestration.

Runs the full RAG pipeline via RagService.invoke() with cache disabled,
then evaluates quality via LLM judge.
"""

import asyncio
import json
import logging
from pathlib import Path

from config import settings
from domain.exceptions import BenchmarkQuestionsNotFound

from infrastructure.benchmark.case_metrics import (
    _compute_retriever_metrics_from_sources as _compute_retriever_metrics_from_sources,
)
from infrastructure.benchmark.case_metrics import (
    _extract_usage_from_response as _extract_usage_from_response,
)

logger = logging.getLogger("default")

EXAMPLE_QUESTIONS = [
    {
        "id": "q1",
        "question": "Какие товары подлежат обязательной маркировке?",
        "expected_answer": None,
        "source_hint": None,
    },
    {
        "id": "q2",
        "question": "Каков порядок электронного документооборота?",
        "expected_answer": None,
        "source_hint": "электронном документе",
    },
]


def load_questions(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        logger.warning("Файл %s не найден — создаю пример test_questions.json", path)
        example_path = Path(path)
        example_path.write_text(json.dumps(EXAMPLE_QUESTIONS, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Отредактируй %s и запусти снова.", path)
        raise BenchmarkQuestionsNotFound(path)

    data = json.loads(p.read_text(encoding="utf-8"))
    logger.info("Загружено вопросов: %d", len(data))
    return data


async def run_benchmark_async(
    questions_path: str,
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
        "questions=%s top_k=%d model=%s judge=%s seed=%s runs=%d",
        questions_path,
        top_k,
        settings.llm_model,
        judge_model,
        seed,
        n_runs,
    )
    questions = await asyncio.to_thread(load_questions, questions_path)
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
