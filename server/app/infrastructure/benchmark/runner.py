"""Benchmark runner — main benchmark orchestration."""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from config import get_setting, settings

from infrastructure.benchmark.judge import (
    get_rag_answer,
    judge_answer,
    judge_answer_async,
)
from infrastructure.benchmark.metrics import (
    compute_context_precision_recall,
    compute_retriever_metrics,
)
from infrastructure.benchmark.persistence import (
    log_question_result,
    log_summary,
    save_results,
)
from infrastructure.benchmark.retrieval import build_llm, retrieve_with_scores_hybrid

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
        sys.exit(0)

    data = json.loads(p.read_text(encoding="utf-8"))
    logger.info("Загружено вопросов: %d", len(data))
    return data


def run_benchmark(
    questions_path: str,
    out_dir: str,
    top_k: int,
    judge_model: str,
    seed: int | None = None,
    n_runs: int = 1,
):
    logger.info("RAG Benchmark")
    logger.info("  questions : %s", questions_path)
    logger.info("  top_k     : %d", top_k)
    logger.info("  provider  : %s", settings.llm_provider)
    logger.info("  rag model : %s", settings.llm_model)
    logger.info("  judge     : %s", judge_model)
    logger.info("  seed      : %s", seed if seed is not None else "none")
    logger.info("  n_runs    : %d", n_runs)
    logger.info("  qdrant    : %s", settings.qdrant_url)

    questions = load_questions(questions_path)

    fetch_k = int(get_setting("rag.retriever_fetch_k"))

    logger.info("Подключаюсь к RAG LLM (%s) ...", settings.llm_model)
    rag_llm = build_llm(settings.llm_model, settings.ollama_base_url, provider=settings.llm_provider)
    if seed is not None:
        rag_llm = rag_llm.with_config({"configurable": {"seed": seed}})

    logger.info("Подключаюсь к LLM-судье (%s) ...", judge_model)
    judge_llm = build_llm(judge_model, settings.ollama_base_url, provider=settings.llm_provider)
    if seed is not None:
        judge_llm = judge_llm.with_config({"configurable": {"seed": seed}})

    logger.info("Прогрев моделей ...")
    rag_llm.invoke("Привет")
    if judge_model != settings.llm_model:
        judge_llm.invoke("Привет")

    all_results: list[dict] = []

    for run_idx in range(1, n_runs + 1):
        if n_runs > 1:
            logger.info("=== Run %d/%d ===", run_idx, n_runs)

        logger.info("Запускаю тесты (параллельные judge-выcalls)...")
        results = []
        total_start = time.time()

        for idx, q in enumerate(questions, 1):
            t_start = time.time()

            docs_with_scores = retrieve_with_scores_hybrid(q["question"], top_k, fetch_k)
            answer = get_rag_answer(rag_llm, docs_with_scores, q["question"])
            retriever_metrics = compute_retriever_metrics(docs_with_scores, q.get("source_hint"))

            context_for_judge = "\n\n---\n\n".join(d.page_content for d, _ in docs_with_scores)
            generator_metrics = judge_answer(
                question=q["question"],
                answer=answer,
                context=context_for_judge,
                expected_answer=q.get("expected_answer"),
            )

            context_metrics = compute_context_precision_recall(
                question=q["question"],
                answer=answer,
                docs_with_scores=docs_with_scores,
            )

            latency = time.time() - t_start

            result = {
                "id": q.get("id", str(idx)),
                "question": q["question"],
                "answer": answer,
                "expected_answer": q.get("expected_answer"),
                "source_hint": q.get("source_hint"),
                "retriever_metrics": retriever_metrics,
                "generator_metrics": generator_metrics,
                "context_metrics": context_metrics,
                "latency_sec": round(latency, 2),
            }
            results.append(result)
            log_question_result(idx, len(questions), q, result)

        total_time = time.time() - total_start
        log_summary(results, total_time)

        for r in results:
            r["run"] = run_idx
        all_results.extend(results)

    if n_runs > 1:
        save_results(all_results, out_dir, model_name=settings.llm_model, run_id="all")

        logger.info("=" * 60)
        logger.info("AGGREGATE SUMMARY (%d runs)", n_runs)
        logger.info("=" * 60)

        from infrastructure.benchmark.metrics import _safe_avg

        question_ids = {r["id"] for r in all_results}
        agg_results = []
        for qid in question_ids:
            q_runs = [r for r in all_results if r["id"] == qid]
            q_questions = [r["question"] for r in q_runs]

            agg = {
                "id": qid,
                "question": q_questions[0] if q_questions else "",
                "n_runs": len(q_runs),
                "retriever_metrics": {
                    "avg_hit_rate": _safe_avg(r["retriever_metrics"]["avg_hit_rate"] for r in q_runs),
                    "avg_mrr": _safe_avg(r["retriever_metrics"]["avg_mrr"] for r in q_runs),
                },
                "generator_metrics": {
                    "faithfulness": _safe_avg(r["generator_metrics"]["faithfulness"] for r in q_runs),
                    "relevancy": _safe_avg(r["generator_metrics"]["relevancy"] for r in q_runs),
                    "correctness": _safe_avg(r["generator_metrics"]["correctness"] for r in q_runs),
                },
                "context_metrics": {
                    "context_precision": _safe_avg(
                        r.get("context_metrics", {}).get("context_precision") for r in q_runs
                    ),
                    "context_recall": _safe_avg(
                        r.get("context_metrics", {}).get("context_recall") for r in q_runs
                    ),
                },
                "latency_sec": _safe_avg(r["latency_sec"] for r in q_runs),
            }
            agg_results.append(agg)

        agg_metrics = {
            "avg_hit_rate": _safe_avg(r["retriever_metrics"]["avg_hit_rate"] for r in agg_results),
            "avg_mrr": _safe_avg(r["retriever_metrics"]["avg_mrr"] for r in agg_results),
            "avg_faithfulness": _safe_avg(r["generator_metrics"]["faithfulness"] for r in agg_results),
            "avg_relevancy": _safe_avg(r["generator_metrics"]["relevancy"] for r in agg_results),
            "avg_correctness": _safe_avg(r["generator_metrics"]["correctness"] for r in agg_results),
            "avg_context_precision": _safe_avg(
                r.get("context_metrics", {}).get("context_precision") for r in agg_results
            ),
            "avg_context_recall": _safe_avg(
                r.get("context_metrics", {}).get("context_recall") for r in agg_results
            ),
            "avg_latency": _safe_avg(r["latency_sec"] for r in agg_results),
        }
        logger.info("  Hit Rate:  %.3f", agg_metrics["avg_hit_rate"])
        logger.info("  MRR:       %.4f", agg_metrics["avg_mrr"])
        logger.info("  Faith:     %.1f/10", agg_metrics["avg_faithfulness"])
        logger.info("  Rel:       %.1f/10", agg_metrics["avg_relevancy"])
        logger.info("  Correct:   %.1f/10", agg_metrics["avg_correctness"])
        logger.info("  Ctx Prec:  %.1f/10", agg_metrics["avg_context_precision"])
        logger.info("  Ctx Rec:   %.1f/10", agg_metrics["avg_context_recall"])
        logger.info("  Latency:   %.1fs", agg_metrics["avg_latency"])
    else:
        save_results(all_results, out_dir, model_name=settings.llm_model)


async def run_benchmark_async(
    questions_path: str,
    out_dir: str,
    top_k: int,
    judge_model: str,
    max_concurrent: int = 4,
    seed: int | None = None,
    n_runs: int = 1,
):
    """Async benchmark with parallel question processing and parallel judge calls."""
    logger.info("RAG Benchmark (async, max_concurrent=%d)", max_concurrent)
    logger.info("  questions : %s", questions_path)
    logger.info("  top_k     : %d", top_k)
    logger.info("  provider  : %s", settings.llm_provider)
    logger.info("  rag model : %s", settings.llm_model)
    logger.info("  judge     : %s", judge_model)
    logger.info("  seed      : %s", seed if seed is not None else "none")
    logger.info("  n_runs    : %d", n_runs)

    questions = load_questions(questions_path)
    fetch_k = int(get_setting("rag.retriever_fetch_k"))

    rag_llm = build_llm(settings.llm_model, settings.ollama_base_url, provider=settings.llm_provider)
    if seed is not None:
        rag_llm = rag_llm.with_config({"configurable": {"seed": seed}})

    judge_llm = build_llm(judge_model, settings.ollama_base_url, provider=settings.llm_provider)
    if seed is not None:
        judge_llm = judge_llm.with_config({"configurable": {"seed": seed}})

    logger.info("Прогрев моделей ...")
    await asyncio.to_thread(rag_llm.invoke, "Привет")
    if judge_model != settings.llm_model:
        await asyncio.to_thread(judge_llm.invoke, "Привет")

    semaphore = asyncio.Semaphore(max_concurrent)

    all_results: list[dict] = []

    for run_idx in range(1, n_runs + 1):
        if n_runs > 1:
            logger.info("=== Run %d/%d ===", run_idx, n_runs)

        async def _process_question(idx: int, q: dict, _run_idx=run_idx) -> dict:
            async with semaphore:
                t_start = time.time()

                docs_with_scores = await asyncio.to_thread(
                    retrieve_with_scores_hybrid, q["question"], top_k, fetch_k
                )
                answer = await asyncio.to_thread(get_rag_answer, rag_llm, docs_with_scores, q["question"])
                retriever_metrics = compute_retriever_metrics(docs_with_scores, q.get("source_hint"))

                context_for_judge = "\n\n---\n\n".join(d.page_content for d, _ in docs_with_scores)
                generator_metrics = await judge_answer_async(
                    question=q["question"],
                    answer=answer,
                    context=context_for_judge,
                    expected_answer=q.get("expected_answer"),
                )

                context_metrics = await asyncio.to_thread(
                    compute_context_precision_recall,
                    q["question"],
                    answer,
                    docs_with_scores,
                )

                latency = time.time() - t_start

                result = {
                    "id": q.get("id", str(idx)),
                    "question": q["question"],
                    "answer": answer,
                    "expected_answer": q.get("expected_answer"),
                    "source_hint": q.get("source_hint"),
                    "retriever_metrics": retriever_metrics,
                    "generator_metrics": generator_metrics,
                    "context_metrics": context_metrics,
                    "latency_sec": round(latency, 2),
                    "run": _run_idx,
                }
                log_question_result(idx, len(questions), q, result)
                return result

        logger.info("Запускаю тесты...")
        total_start = time.time()

        tasks = [_process_question(idx, q) for idx, q in enumerate(questions, 1)]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - total_start
        log_summary(list(results), total_time)
        all_results.extend(results)

    save_results(all_results, out_dir, model_name=settings.llm_model, run_id="all" if n_runs > 1 else "")
