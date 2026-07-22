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

import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from config import settings
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore

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
    logger.info("Загружаю эмбеддинг-модель %s ...", settings.embed_model)
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
    )
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


def judge_answer(
    judge_llm: ChatOllama,
    question: str,
    answer: str,
    context: str,
    expected_answer: str | None = None,
) -> dict:
    scores = {}

    raw = judge_llm.invoke(
        FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer)
    ).content
    scores["faithfulness"], scores["faithfulness_reason"] = parse_judge_response(raw, "faithfulness")

    raw = judge_llm.invoke(RELEVANCY_PROMPT.format(question=question, answer=answer)).content
    scores["relevancy"], scores["relevancy_reason"] = parse_judge_response(raw, "relevancy")

    if expected_answer:
        raw = judge_llm.invoke(
            CORRECTNESS_PROMPT.format(question=question, expected=expected_answer, answer=answer)
        ).content
        scores["correctness"], scores["correctness_reason"] = parse_judge_response(raw, "correctness")
    else:
        scores["correctness"] = None
        scores["correctness_reason"] = "Эталонный ответ не задан"

    return scores


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
# Итоговый отчёт
# ---------------------------------------------------------------------------


def log_summary(results: list[dict], total_time: float):
    n = len(results)
    logger.info("=" * 60)
    logger.info("ИТОГОВЫЙ ОТЧЁТ  (%d вопросов, %.1fs)", n, total_time)
    logger.info("=" * 60)

    hit_rates = [
        r["retriever_metrics"]["hit_rate"] for r in results if r["retriever_metrics"]["hit_rate"] is not None
    ]
    mrrs = [r["retriever_metrics"]["mrr"] for r in results if r["retriever_metrics"]["mrr"] is not None]
    sims = [r["retriever_metrics"]["avg_similarity"] for r in results]

    logger.info("Retriever:")
    if hit_rates:
        avg_hr = sum(hit_rates) / len(hit_rates)
        avg_mrr = sum(mrrs) / len(mrrs)
        logger.info(
            "  Hit Rate:        %.1f/10  (%.0f%% вопросов нашли нужный источник)", avg_hr * 10, avg_hr * 100
        )
        logger.info("  MRR:             %.3f  (1.0 = нужный чанк всегда первый)", avg_mrr)
    logger.info("  Avg Similarity:  %.3f", sum(sims) / len(sims))

    faiths = [r["generator_metrics"]["faithfulness"] for r in results]
    rels = [r["generator_metrics"]["relevancy"] for r in results]
    corrs = [
        r["generator_metrics"]["correctness"]
        for r in results
        if r["generator_metrics"]["correctness"] is not None
    ]

    logger.info("Generator:")
    logger.info(
        "  Faithfulness:    %.1f/10  (достоверность — нет ли выдуманных фактов)", sum(faiths) / len(faiths)
    )
    logger.info("  Relevancy:       %.1f/10  (ответ по существу вопроса)", sum(rels) / len(rels))
    if corrs:
        logger.info("  Correctness:     %.1f/10  (совпадение с эталоном)", sum(corrs) / len(corrs))

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


def save_results(results: list[dict], out_dir: str, model_name: str = ""):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = f"_{_sanitize_model_name(model_name)}" if model_name else ""

    json_path = out / f"benchmark_{ts}{model_tag}.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out / f"benchmark_{ts}{model_tag}.csv"
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


# ---------------------------------------------------------------------------
# Главный цикл
# ---------------------------------------------------------------------------


def run_benchmark(
    questions_path: str,
    out_dir: str,
    top_k: int,
    judge_model: str,
):
    logger.info("RAG Benchmark")
    logger.info("  questions : %s", questions_path)
    logger.info("  top_k     : %d", top_k)
    logger.info("  rag model : %s", settings.llm_model)
    logger.info("  judge     : %s", judge_model)
    logger.info("  qdrant    : %s", settings.qdrant_url)

    questions = load_questions(questions_path)

    retriever, vs = build_retriever(top_k)

    logger.info("Подключаюсь к RAG LLM (%s) ...", settings.llm_model)
    rag_llm = build_llm(settings.llm_model, settings.ollama_base_url)

    logger.info("Подключаюсь к LLM-судье (%s) ...", judge_model)
    judge_llm = build_llm(judge_model, settings.ollama_base_url)

    logger.info("Прогрев моделей ...")
    rag_llm.invoke("Привет")
    if judge_model != settings.llm_model:
        judge_llm.invoke("Привет")

    logger.info("Запускаю тесты...")
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
    save_results(results, out_dir, model_name=settings.llm_model)
