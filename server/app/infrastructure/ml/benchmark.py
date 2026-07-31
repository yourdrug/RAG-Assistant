"""
benchmark.py — оценка качества RAG-системы.

Что измеряет:
  Retriever:
    hit_rate       — есть ли среди top-k чанков хотя бы один с правильным источником
    mrr            — Mean Reciprocal Rank (насколько высоко стоит правильный чанк)
    avg_similarity — средний similarity score найденных чанков

  Generator (LLM-судья через Ollama):
    faithfulness   — ответ основан на контексте или модель придумала? (0-10)
    relevancy      — ответ по существу вопроса? (0-10)
    correctness    — совпадает с эталонным ответом? (0-10, только если задан expected_answer)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from config import settings
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore

from infrastructure.clients import get_vector_store

logger = logging.getLogger("default")


# ---------------------------------------------------------------------------
# Загрузка тестовых вопросов
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Компоненты RAG
# ---------------------------------------------------------------------------


def build_retriever(top_k: int):
    logger.info("Using cached embedding model %s ...", settings.embed_model)
    vs = get_vector_store()
    return vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": 0.0},
    ), vs


def retrieve_with_scores(vs: QdrantVectorStore, question: str, top_k: int) -> list[tuple[Document, float]]:
    results = vs.similarity_search_with_score(question, k=top_k)
    return results


def build_llm(model: str, base_url: str) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# RAG: получить ответ
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """\
Ты — корпоративный ассистент. Отвечай только на основе предоставленного контекста.
Если ответа в контексте нет — напиши ровно: "Информация в документах не найдена."
Не придумывай факты.

Контекст:
{context}

Вопрос: {question}
"""


def get_rag_answer(llm: ChatOllama, docs_with_scores: list[tuple[Document, float]], question: str) -> str:
    context = "\n\n---\n\n".join(
        f"[Источник: {d.metadata.get('filename', 'unknown')}]\n{d.page_content}" for d, _ in docs_with_scores
    )
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content.strip()


# ---------------------------------------------------------------------------
# LLM-судья: оценки
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """\
Ты — строгий эксперт по оценке качества ответов AI-ассистентов.

Контекст из документов:
{context}

Вопрос: {question}

Ответ ассистента: {answer}

Задача: оцени FAITHFULNESS (достоверность) — насколько ответ основан на предоставленном контексте.
Ответ полностью из контекста = 10. Ответ содержит выдуманные факты = 0.

Ответь СТРОГО в формате JSON (только JSON, без пояснений):
{{"score": <число от 0 до 10>, "reason": "<одно предложение>"}}
"""

RELEVANCY_PROMPT = """\
Ты — строгий эксперт по оценке качества ответов AI-ассистентов.

Вопрос: {question}

Ответ ассистента: {answer}

Задача: оцени RELEVANCY (релевантность) — насколько ответ отвечает на поставленный вопрос.
Точный полный ответ = 10. Ответ не по теме = 0.

Ответь СТРОГО в формате JSON (только JSON, без пояснений):
{{"score": <число от 0 до 10>, "reason": "<одно предложение>"}}
"""

CORRECTNESS_PROMPT = """\
Ты — строгий эксперт по оценке качества ответов AI-ассистентов.

Вопрос: {question}

Эталонный ответ: {expected}

Ответ ассистента: {answer}

Задача: оцени CORRECTNESS (правильность) — насколько ответ совпадает по смыслу с эталонным.
Полное совпадение по смыслу = 10. Противоречит эталону = 0.

Ответь СТРОГО в формате JSON (только JSON, без пояснений):
{{"score": <число от 0 до 10>, "reason": "<одно предложение>"}}
"""


def parse_judge_response(raw: str, metric: str) -> tuple[float, str]:
    match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if not match:
        return 0.0, f"[Ошибка парсинга ответа судьи: {raw[:100]}]"
    try:
        data = json.loads(match.group())
        score = float(data.get("score", 0))
        score = max(0.0, min(10.0, score))
        reason = str(data.get("reason", ""))
        return score, reason
    except (json.JSONDecodeError, ValueError) as e:
        return 0.0, f"[JSON parse error: {e}]"


async def judge_answer_async(
    judge_llm: ChatOllama,
    question: str,
    answer: str,
    context: str,
    expected_answer: str | None = None,
) -> dict:
    """Judge answer quality with 3 parallel LLM calls (faithfulness, relevancy, correctness)."""

    async def _judge_one(prompt: str) -> str:
        return await asyncio.to_thread(judge_llm.invoke, prompt).content

    prompts = {
        "faithfulness": FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer),
        "relevancy": RELEVANCY_PROMPT.format(question=question, answer=answer),
    }
    if expected_answer:
        prompts["correctness"] = CORRECTNESS_PROMPT.format(
            question=question, expected=expected_answer, answer=answer
        )

    keys = list(prompts.keys())
    raw_results = await asyncio.gather(*[_judge_one(prompts[k]) for k in keys])

    scores = {}
    for key, raw in zip(keys, raw_results):
        score, reason = parse_judge_response(raw, key)
        scores[key] = score
        scores[f"{key}_reason"] = reason

    if "correctness" not in scores:
        scores["correctness"] = None
        scores["correctness_reason"] = "Эталонный ответ не задан"

    return scores


def judge_answer(
    judge_llm: ChatOllama,
    question: str,
    answer: str,
    context: str,
    expected_answer: str | None = None,
) -> dict:
    """Sync wrapper for judge_answer_async — runs in event loop or standalone."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in async context — use to_thread for blocking invoke
        scores = {}

        def _invoke(prompt: str) -> str:
            return judge_llm.invoke(prompt).content

        prompts = {
            "faithfulness": FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer),
            "relevancy": RELEVANCY_PROMPT.format(question=question, answer=answer),
        }
        if expected_answer:
            prompts["correctness"] = CORRECTNESS_PROMPT.format(
                question=question, expected=expected_answer, answer=answer
            )

        # Sequential fallback when already in running loop
        for key, prompt in prompts.items():
            raw = judge_llm.invoke(prompt).content
            score, reason = parse_judge_response(raw, key)
            scores[key] = score
            scores[f"{key}_reason"] = reason

        if "correctness" not in scores:
            scores["correctness"] = None
            scores["correctness_reason"] = "Эталонный ответ не задан"
        return scores
    else:
        return asyncio.run(
            judge_answer_async(judge_llm, question, answer, context, expected_answer)
        )


# ---------------------------------------------------------------------------
# Retriever метрики
# ---------------------------------------------------------------------------


def compute_retriever_metrics(
    docs_with_scores: list[tuple[Document, float]],
    source_hint: str | None,
) -> dict:
    scores_list = [s for _, s in docs_with_scores]
    avg_sim = sum(scores_list) / len(scores_list) if scores_list else 0.0

    if source_hint is None:
        return {
            "hit_rate": None,
            "mrr": None,
            "avg_similarity": round(avg_sim, 4),
            "retrieved_sources": [d.metadata.get("filename", "?") for d, _ in docs_with_scores],
        }

    hit_rate = 0
    mrr = 0.0
    for rank, (doc, _) in enumerate(docs_with_scores, 1):
        filename = doc.metadata.get("filename", "") or doc.metadata.get("source", "")
        if source_hint.lower() in filename.lower():
            hit_rate = 1
            if mrr == 0.0:
                mrr = 1.0 / rank
            break

    return {
        "hit_rate": hit_rate,
        "mrr": round(mrr, 4),
        "avg_similarity": round(avg_sim, 4),
        "retrieved_sources": [d.metadata.get("filename", "?") for d, _ in docs_with_scores],
    }


# ---------------------------------------------------------------------------
# Форматирование прогресса
# ---------------------------------------------------------------------------


def log_question_result(idx: int, total: int, q: dict, result: dict):
    logger.info("[%d/%d] %s", idx, total, q["question"])

    rm = result["retriever_metrics"]
    sim_str = f"avg_sim={rm['avg_similarity']:.3f}"
    if rm["hit_rate"] is not None:
        hr_str = "hit" if rm["hit_rate"] else "miss"
        mrr_str = f"mrr={rm['mrr']:.2f}"
        logger.info("  Retriever: %s  %s  %s", hr_str, mrr_str, sim_str)
    else:
        logger.info("  Retriever: %s  (source_hint не задан)", sim_str)

    src_list = ", ".join(result["retriever_metrics"]["retrieved_sources"][:3])
    logger.info("  Источники: %s", src_list)

    gm = result["generator_metrics"]
    logger.info("  Faithfulness: %.1f/10  — %s", gm["faithfulness"], gm["faithfulness_reason"])
    logger.info("  Relevancy:    %.1f/10  — %s", gm["relevancy"], gm["relevancy_reason"])
    if gm["correctness"] is not None:
        logger.info("  Correctness:  %.1f/10  — %s", gm["correctness"], gm["correctness_reason"])

    answer_preview = result["answer"][:200].replace("\n", " ")
    if len(result["answer"]) > 200:
        answer_preview += "..."
    logger.info("  Ответ: %s", answer_preview)
    logger.info("  Время: %.1fs", result["latency_sec"])


# ---------------------------------------------------------------------------
# Summary metrics (shared by log_summary, save_results, BenchmarkService)
# ---------------------------------------------------------------------------


def compute_summary_metrics(results: list[dict]) -> dict:
    """Extract aggregated metrics from benchmark results."""
    faiths = [r["generator_metrics"]["faithfulness"] for r in results]
    rels = [r["generator_metrics"]["relevancy"] for r in results]
    corrs = [
        r["generator_metrics"]["correctness"]
        for r in results
        if r["generator_metrics"]["correctness"] is not None
    ]
    hit_rates = [
        r["retriever_metrics"]["hit_rate"] for r in results if r["retriever_metrics"]["hit_rate"] is not None
    ]
    mrrs = [r["retriever_metrics"]["mrr"] for r in results if r["retriever_metrics"]["mrr"] is not None]
    sims = [r["retriever_metrics"]["avg_similarity"] for r in results]

    return {
        "total_questions": len(results),
        "total_time_sec": round(sum(r["latency_sec"] for r in results), 1),
        "hit_rate": round(sum(hit_rates) / len(hit_rates), 3) if hit_rates else None,
        "avg_mrr": round(sum(mrrs) / len(mrrs), 3) if mrrs else None,
        "avg_faithfulness": round(sum(faiths) / len(faiths), 1) if faiths else None,
        "avg_relevancy": round(sum(rels) / len(rels), 1) if rels else None,
        "avg_correctness": round(sum(corrs) / len(corrs), 1) if corrs else None,
        "avg_similarity": round(sum(sims) / len(sims), 3) if sims else 0,
    }


# ---------------------------------------------------------------------------
# Итоговый отчёт
# ---------------------------------------------------------------------------


def log_summary(results: list[dict], total_time: float):
    n = len(results)
    logger.info("=" * 60)
    logger.info("ИТОГОВЫЙ ОТЧЁТ  (%d вопросов, %.1fs)", n, total_time)
    logger.info("=" * 60)

    m = compute_summary_metrics(results)

    logger.info("Retriever:")
    if m["hit_rate"] is not None:
        logger.info(
            "  Hit Rate:        %.1f/10  (%.0f%% вопросов нашли нужный источник)", m["hit_rate"] * 10, m["hit_rate"] * 100
        )
        logger.info("  MRR:             %.3f  (1.0 = нужный чанк всегда первый)", m["avg_mrr"])
    logger.info("  Avg Similarity:  %.3f", m["avg_similarity"])

    logger.info("Generator:")
    logger.info(
        "  Faithfulness:    %.1f/10  (достоверность — нет ли выдуманных фактов)", m["avg_faithfulness"]
    )
    logger.info("  Relevancy:       %.1f/10  (ответ по существу вопроса)", m["avg_relevancy"])
    if m["avg_correctness"] is not None:
        logger.info("  Correctness:     %.1f/10  (совпадение с эталоном)", m["avg_correctness"])

    bad = [
        r
        for r in results
        if r["generator_metrics"]["faithfulness"] < 5 or r["generator_metrics"]["relevancy"] < 5
    ]
    if bad:
        logger.warning("Проблемные вопросы (%d):", len(bad))
        for r in bad:
            gm = r["generator_metrics"]
            logger.warning("  [%s] %s", r["id"], r["question"][:60])
            logger.warning("        faith=%.1f  rel=%.1f", gm["faithfulness"], gm["relevancy"])
            logger.warning("        %s", gm["faithfulness_reason"])

    logger.info("Время: %.1fs  (%.1fs на вопрос)", total_time, total_time / n)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Сохранение результатов
# ---------------------------------------------------------------------------


def _sanitize_model_name(model: str) -> str:
    """Replace characters invalid in filenames (e.g. ':') with underscores."""
    return re.sub(r'[\\/:*?"<>|]', "_", model)


def save_results(results: list[dict], out_dir: str, model_name: str = "", run_id: str = ""):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = f"_{_sanitize_model_name(model_name)}" if model_name else ""
    run_tag = f"_{run_id}" if run_id else ""

    json_path = out / f"benchmark_{ts}{model_tag}{run_tag}.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out / f"benchmark_{ts}{model_tag}{run_tag}.csv"
    rows = ["id,question,faithfulness,relevancy,correctness,hit_rate,mrr,avg_sim,latency_sec"]
    for r in results:
        gm = r["generator_metrics"]
        rm = r["retriever_metrics"]
        rows.append(
            ",".join(
                [
                    str(r["id"]),
                    f'"{r["question"]}"',
                    str(gm["faithfulness"]),
                    str(gm["relevancy"]),
                    str(gm["correctness"] if gm["correctness"] is not None else ""),
                    str(rm["hit_rate"] if rm["hit_rate"] is not None else ""),
                    str(rm["mrr"] if rm["mrr"] is not None else ""),
                    str(rm["avg_similarity"]),
                    str(round(r["latency_sec"], 2)),
                ]
            )
        )
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    logger.info("Результаты сохранены:")
    logger.info("  JSON: %s", json_path)
    logger.info("  CSV:  %s", csv_path)

    # Append to history for trend tracking
    from infrastructure.ml.benchmark_history import save_summary_to_history

    summary = compute_summary_metrics(results)

    config = {
        "top_k": settings.retriever_top_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embed_model": settings.embed_model,
        "llm_model": model_name or settings.llm_model,
        "hybrid_enabled": settings.hybrid_enabled,
        "dense_weight": settings.dense_weight,
        "sparse_weight": settings.sparse_weight,
        "rrf_k": settings.rrf_k,
    }

    save_summary_to_history(summary, config, settings.data_dir)


# ---------------------------------------------------------------------------
# Grid search helpers (retrieval-only, no LLM)
# ---------------------------------------------------------------------------


def compute_retrieval_metrics_grid(
    dense_by_hash: dict,
    sparse_results: list[tuple[str, float]],
    questions: list[dict],
    candidates_by_hash: dict,
    top_k: int,
    rrf_k: int,
    dense_weight: float,
    sparse_weight: float,
) -> dict:
    """Compute retrieval metrics for a single RRF config in-memory (no LLM/Qdrant calls)."""
    from infrastructure.ml.hybrid import rrf_merge

    merged_hashes = rrf_merge(
        [(h, s) for h, s, _ in []] if not dense_by_hash else list(dense_by_hash.values()),
        sparse_results,
        k=rrf_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )

    # Deduplicate and take top_k
    seen = set()
    top_hashes = []
    for h in merged_hashes:
        if h not in seen:
            seen.add(h)
            top_hashes.append(h)
            if len(top_hashes) >= top_k:
                break

    # Compute metrics for each question
    hit_rates = []
    mrrs = []

    for q in questions:
        source_hint = q.get("source_hint")
        if source_hint is None:
            continue

        hit = 0
        mrr = 0.0
        for rank, h in enumerate(top_hashes, 1):
            doc = candidates_by_hash.get(h)
            if doc is None:
                continue
            filename = doc.metadata.get("filename", "") or doc.metadata.get("source", "")
            if source_hint.lower() in filename.lower():
                hit = 1
                if mrr == 0.0:
                    mrr = 1.0 / rank
                break

        hit_rates.append(hit)
        mrrs.append(mrr)

    avg_hr = sum(hit_rates) / len(hit_rates) if hit_rates else 0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0

    return {
        "avg_hit_rate": round(avg_hr, 3),
        "avg_mrr": round(avg_mrr, 4),
    }


# ---------------------------------------------------------------------------
# Главный цикл
# ---------------------------------------------------------------------------


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
    logger.info("  rag model : %s", settings.llm_model)
    logger.info("  judge     : %s", judge_model)
    logger.info("  seed      : %s", seed if seed is not None else "none")
    logger.info("  n_runs    : %d", n_runs)
    logger.info("  qdrant    : %s", settings.qdrant_url)

    questions = load_questions(questions_path)

    retriever, vs = build_retriever(top_k)

    logger.info("Подключаюсь к RAG LLM (%s) ...", settings.llm_model)
    rag_llm = build_llm(settings.llm_model, settings.ollama_base_url)
    if seed is not None:
        rag_llm = rag_llm.with_config({"configurable": {"seed": seed}})

    logger.info("Подключаюсь к LLM-судье (%s) ...", judge_model)
    judge_llm = build_llm(judge_model, settings.ollama_base_url)
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

            docs_with_scores = retrieve_with_scores(vs, q["question"], top_k)
            answer = get_rag_answer(rag_llm, docs_with_scores, q["question"])
            retriever_metrics = compute_retriever_metrics(docs_with_scores, q.get("source_hint"))

            context_for_judge = "\n\n---\n\n".join(d.page_content for d, _ in docs_with_scores)
            generator_metrics = judge_answer(
                judge_llm,
                question=q["question"],
                answer=answer,
                context=context_for_judge,
                expected_answer=q.get("expected_answer"),
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
                "latency_sec": round(latency, 2),
            }
            results.append(result)
            log_question_result(idx, len(questions), q, result)

        total_time = time.time() - total_start
        log_summary(results, total_time)

        # Tag with run index
        for r in results:
            r["run"] = run_idx
        all_results.extend(results)

    # Save all runs (for multi-run, we save aggregate + per-run)
    if n_runs > 1:
        save_results(all_results, out_dir, model_name=settings.llm_model, run_id="all")

        # Compute and log aggregate summary
        logger.info("=" * 60)
        logger.info("AGGREGATE SUMMARY (%d runs)", n_runs)
        logger.info("=" * 60)

        question_ids = list(set(r["id"] for r in all_results))
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
                "latency_sec": _safe_avg(r["latency_sec"] for r in q_runs),
            }
            agg_results.append(agg)

        agg_metrics = {
            "avg_hit_rate": _safe_avg(r["retriever_metrics"]["avg_hit_rate"] for r in agg_results),
            "avg_mrr": _safe_avg(r["retriever_metrics"]["avg_mrr"] for r in agg_results),
            "avg_faithfulness": _safe_avg(r["generator_metrics"]["faithfulness"] for r in agg_results),
            "avg_relevancy": _safe_avg(r["generator_metrics"]["relevancy"] for r in agg_results),
            "avg_correctness": _safe_avg(r["generator_metrics"]["correctness"] for r in agg_results),
            "avg_latency": _safe_avg(r["latency_sec"] for r in agg_results),
        }
        logger.info("  Hit Rate:  %.3f", agg_metrics["avg_hit_rate"])
        logger.info("  MRR:       %.4f", agg_metrics["avg_mrr"])
        logger.info("  Faith:     %.1f/10", agg_metrics["avg_faithfulness"])
        logger.info("  Rel:       %.1f/10", agg_metrics["avg_relevancy"])
        logger.info("  Correct:   %.1f/10", agg_metrics["avg_correctness"])
        logger.info("  Latency:   %.1fs", agg_metrics["avg_latency"])
    else:
        save_results(all_results, out_dir, model_name=settings.llm_model)


def _safe_avg(vals) -> float:
    vals = list(vals)
    return round(sum(vals) / len(vals), 4) if vals else 0


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
    logger.info("  rag model : %s", settings.llm_model)
    logger.info("  judge     : %s", judge_model)
    logger.info("  seed      : %s", seed if seed is not None else "none")
    logger.info("  n_runs    : %d", n_runs)

    questions = load_questions(questions_path)
    retriever, vs = build_retriever(top_k)

    rag_llm = build_llm(settings.llm_model, settings.ollama_base_url)
    if seed is not None:
        rag_llm = rag_llm.with_config({"configurable": {"seed": seed}})

    judge_llm = build_llm(judge_model, settings.ollama_base_url)
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

        async def _process_question(idx: int, q: dict) -> dict:
            async with semaphore:
                t_start = time.time()

                docs_with_scores = await asyncio.to_thread(retrieve_with_scores, vs, q["question"], top_k)
                answer = await asyncio.to_thread(get_rag_answer, rag_llm, docs_with_scores, q["question"])
                retriever_metrics = compute_retriever_metrics(docs_with_scores, q.get("source_hint"))

                context_for_judge = "\n\n---\n\n".join(d.page_content for d, _ in docs_with_scores)
                generator_metrics = await judge_answer_async(
                    judge_llm,
                    question=q["question"],
                    answer=answer,
                    context=context_for_judge,
                    expected_answer=q.get("expected_answer"),
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
                    "latency_sec": round(latency, 2),
                    "run": run_idx,
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

    save_results(all_results, out_dir, model_name=settings.llm_model, run_id="all" if n_runs > 1 else None)
