"""RAG quality benchmark: retrieval hit-rate, MRR, and LLM-judge scores.

Retriever metrics:
    hit_rate       -- at least one correct-source chunk in top-k
    mrr            -- Mean Reciprocal Rank of the correct chunk
    avg_similarity -- mean cosine similarity of retrieved chunks

Generator metrics (LLM-judge via Ollama):
    faithfulness   -- answer grounded in context (0-10)
    relevancy      -- answer addresses the question (0-10)
    correctness    -- matches expected answer (0-10, only if expected_answer provided)
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
from domain.services.rag_policy import build_system_prompt
from domain.value_objects.llm_provider import Breadth, LLMProvider
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from infrastructure.ml.benchmark_history import save_summary_to_history
from infrastructure.ml.factories import (
    create_embeddings,
    create_qdrant_client,
    create_reranker,
    load_bm25_index,
)
from infrastructure.ml.hybrid import content_hash, rrf_merge
from infrastructure.ml.instructor_client import create_instructor_client
from infrastructure.ml.llm_schemas import JudgeScore
from infrastructure.ml.rag import deduplicate_docs

logger = logging.getLogger("default")

JUDGE_MAX_RETRIES = 3
JUDGE_RETRY_DELAY = 5.0

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


def _search_dense(
    question: str, fetch_k: int
) -> tuple[list[tuple[Document, float]], dict[str, tuple[float, Document]]]:
    """Dense search via Qdrant client. Returns (dense_docs, dense_by_hash)."""
    client = create_qdrant_client()
    embeddings = create_embeddings()
    query_vector = embeddings.embed_query_sync(question)
    dense_results = client.search(
        collection_name=settings.collection_name,
        query_vector=query_vector,
        limit=fetch_k,
    )

    dense_docs: list[tuple[Document, float]] = []
    dense_by_hash: dict[str, tuple[float, Document]] = {}
    for point in dense_results:
        payload = point.payload or {}
        page_content = payload.get("page_content", "")
        metadata = payload.get("metadata", {})
        h = metadata.get("content_hash") or content_hash(page_content)
        doc = Document(page_content=page_content, metadata=metadata)
        dense_docs.append((doc, point.score))
        dense_by_hash[h] = (point.score, doc)
    return dense_docs, dense_by_hash


def _search_sparse(question: str, fetch_k: int) -> list[tuple[str, float]]:
    """BM25 sparse search via loaded index."""
    bm25_index = load_bm25_index()
    if bm25_index is not None:
        return bm25_index.search_with_hashes(question, fetch_k)
    return []


def _merge_and_dedup(
    dense_by_hash: dict[str, tuple[float, Document]],
    sparse_results: list[tuple[str, float]],
    fetch_k: int,
) -> list[Document]:
    """RRF merge + deduplication. Returns deduplicated candidate docs."""
    if sparse_results:
        merged_hashes = rrf_merge(
            [(k, v[0]) for k, v in dense_by_hash.items()],
            sparse_results,
            k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            sparse_weight=settings.sparse_weight,
        )
    else:
        merged_hashes = [h for h, _ in [(k, v[0]) for k, v in dense_by_hash.items()]]

    seen = set()
    candidate_docs: list[Document] = []
    for h in merged_hashes:
        if h in seen:
            continue
        seen.add(h)
        if h in dense_by_hash:
            candidate_docs.append(dense_by_hash[h][1])
        if len(candidate_docs) >= fetch_k:
            break

    return deduplicate_docs(candidate_docs)


def _apply_rerank_filters(
    ranked: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """Apply min_score and score_gap_ratio filters to ranked results."""
    if settings.rerank_min_score is not None:
        ranked = [(d, s) for d, s in ranked if s >= settings.rerank_min_score]

    if settings.rerank_score_gap_ratio is not None and ranked:
        top_score = ranked[0][1]
        cutoff = top_score * settings.rerank_score_gap_ratio
        ranked = [(d, s) for d, s in ranked if s >= cutoff]

    return ranked


def retrieve_with_scores_hybrid(question: str, top_k: int, fetch_k: int) -> list[tuple[Document, float]]:
    """Retrieve using the production hybrid pipeline: dense + BM25 + RRF + reranker.

    Mirrors rag_service._run_hybrid_search + rerank_documents to produce
    the same quality results as the live system.
    """
    _, dense_by_hash = _search_dense(question, fetch_k)
    sparse_results = _search_sparse(question, fetch_k)
    candidate_docs = _merge_and_dedup(dense_by_hash, sparse_results, fetch_k)

    if not candidate_docs:
        return []

    reranker = create_reranker()
    pairs = []
    for doc in candidate_docs:
        source = doc.metadata.get("source", "")
        filename = doc.metadata.get("filename", "")
        doc_name = filename or (Path(source).name if source else "")
        content_with_prefix = f"[{doc_name}] {doc.page_content}" if doc_name else doc.page_content
        pairs.append((question, content_with_prefix))

    scores = reranker.predict_sync(pairs)
    ranked = sorted(zip(candidate_docs, scores, strict=False), key=lambda x: x[1], reverse=True)[:top_k]

    return _apply_rerank_filters(ranked)


def build_llm(model: str, base_url: str, provider: str = LLMProvider.OLLAMA):
    """Build LLM instance based on provider."""
    if provider == LLMProvider.OPENROUTER:
        return ChatOpenAI(
            model=model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=0.0,
        )
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# RAG: получить ответ
# ---------------------------------------------------------------------------

ANSWER_PROMPT_TEMPLATE = """\
{system_prompt}

Вопрос: {question}
"""


def get_rag_answer(llm: ChatOllama, docs_with_scores: list[tuple[Document, float]], question: str) -> str:
    """Generate answer using the production system prompt and formatted context."""
    system_prompt = build_system_prompt(breadth=Breadth.NARROW.value)

    from infrastructure.ml.rag import format_docs

    context = format_docs(docs_with_scores, max_context_tokens=4000)
    prompt_text = ANSWER_PROMPT_TEMPLATE.format(system_prompt=system_prompt, question=question)
    # Append context manually to avoid template conflicts
    full_prompt = (
        f"{prompt_text}\n\nКонтекст из документов:\n<<DOCUMENT_CONTEXT>>\n{context}\n<<END_DOCUMENT_CONTEXT>>"
    )

    last_exc: Exception | None = None
    for attempt in range(1, JUDGE_MAX_RETRIES + 1):
        try:
            response = llm.invoke(full_prompt)
            return response.content.strip()
        except Exception as exc:
            last_exc = exc
            if attempt < JUDGE_MAX_RETRIES:
                delay = JUDGE_RETRY_DELAY * attempt
                logger.warning(
                    "RAG LLM invoke failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    JUDGE_MAX_RETRIES,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error("RAG LLM invoke failed after %d attempts: %s", JUDGE_MAX_RETRIES, exc)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LLM-судья: оценки (structured output via instructor + tenacity)
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

ВАЖНО: Если ответ — «Информация не найдена в документах» или «Информация в документах не найдена»,
это означает, что ассистент корректно определил отсутствие информации.
В этом случае:
- Если информация действительно отсутствует в документах — оцени как 7-10 (ассистент корректноresponded).
- Если информация ЕСТЬ в документах, но ассистент её не нашёл — оцени как 0-3 (ретривер не справился).

Точный полный ответ = 10. Ответ не по теме = 0. «Не найдена» при отсутствии информации = 7-10.

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

CONTEXT_PRECISION_PROMPT = """\
Ты — эксперт по оценке качества поиска (retrieval) в RAG-системе.

Вопрос: {question}

Релевантный ответ (для справки): {answer}

Ретривированные документы (по порядку ретривера):
{context}

Задача: оцени CONTEXT PRECISION — сколько из ретривированных документов
действительно релевантны для ответа на вопрос.
Все релевантны = 10. Ни один не релевантен = 0.
Считай количество релевантных документов и дели на общее число.

Ответь СТРОГО в формате JSON (только JSON, без пояснений):
{{"score": <число от 0 до 10>, "reason": "<одно предложение>"}}
"""

CONTEXT_RECALL_PROMPT = """\
Ты — эксперт по оценке качества поиска (retrieval) в RAG-системе.

Вопрос: {question}

Релевантный ответ (для справки): {answer}

Ретривированные документы:
{context}

Задача: оцени CONTEXT RECALL — какая доля информации, необходимой для ответа
на вопрос, присутствует в ретривированных документах.
Вся необходимая информация есть = 10. Ничего нет = 0.

Ответь СТРОГО в формате JSON (только JSON, без пояснений):
{{"score": <число от 0 до 10>, "reason": "<одно предложение>"}}
"""


def _get_judge_client(model: str):
    """Create an instructor-wrapped client for the judge model."""
    if settings.llm_provider == LLMProvider.OPENROUTER:
        return create_instructor_client(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            model=model,
        )
    return create_instructor_client(
        base_url=f"{settings.ollama_base_url}/v1",
        api_key="ollama",
        model=model,
    )


def _judge_with_structured_output(
    client,
    prompt: str,
    model: str,
) -> JudgeScore:
    """Call judge LLM with structured output + tenacity retry."""
    max_retries = JUDGE_MAX_RETRIES

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def _call() -> JudgeScore:
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=JudgeScore,
            max_retries=3,
        )

    return _call()


def judge_answer(
    judge_llm: ChatOllama,
    question: str,
    answer: str,
    context: str,
    expected_answer: str | None = None,
) -> dict:
    """Run judge LLM synchronously with structured output (instructor + tenacity)."""
    client = _get_judge_client(settings.llm_model if settings.llm_provider == LLMProvider.OLLAMA else "")
    model = settings.llm_model if settings.llm_provider == LLMProvider.OLLAMA else settings.openrouter_model

    prompts = {
        "faithfulness": FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer),
        "relevancy": RELEVANCY_PROMPT.format(question=question, answer=answer),
    }
    if expected_answer:
        prompts["correctness"] = CORRECTNESS_PROMPT.format(
            question=question, expected=expected_answer, answer=answer
        )

    scores = {}
    for key, prompt in prompts.items():
        try:
            result = _judge_with_structured_output(client, prompt, model)
            scores[key] = max(0.0, min(10.0, result.score))
            scores[f"{key}_reason"] = result.reason
        except Exception as exc:
            logger.warning("Judge structured output failed for %s: %s — falling back to 0.0", key, exc)
            scores[key] = 0.0
            scores[f"{key}_reason"] = f"[Ошибка вызова судьи: {exc}]"

    if "correctness" not in scores:
        scores["correctness"] = None
        scores["correctness_reason"] = "Эталонный ответ не задан"
    return scores


async def judge_answer_async(
    judge_llm: ChatOllama,
    question: str,
    answer: str,
    context: str,
    expected_answer: str | None = None,
) -> dict:
    """Judge answer quality with structured output (instructor + tenacity)."""
    client = _get_judge_client(settings.llm_model if settings.llm_provider == LLMProvider.OLLAMA else "")
    model = settings.llm_model if settings.llm_provider == LLMProvider.OLLAMA else settings.openrouter_model

    prompts = {
        "faithfulness": FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer),
        "relevancy": RELEVANCY_PROMPT.format(question=question, answer=answer),
    }
    if expected_answer:
        prompts["correctness"] = CORRECTNESS_PROMPT.format(
            question=question, expected=expected_answer, answer=answer
        )

    async def _judge_one(prompt: str) -> JudgeScore:
        import asyncio as _asyncio

        for attempt in range(1, JUDGE_MAX_RETRIES + 1):
            try:
                result = await _asyncio.to_thread(
                    lambda p=prompt: client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": p}],
                        response_model=JudgeScore,
                        max_retries=3,
                    )
                )
                return result
            except Exception as exc:
                if attempt == JUDGE_MAX_RETRIES:
                    raise
                delay = JUDGE_RETRY_DELAY * attempt
                logger.warning(
                    "Async judge invoke failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    JUDGE_MAX_RETRIES,
                    exc,
                    delay,
                )
                await _asyncio.sleep(delay)
        raise RuntimeError("Unreachable")

    keys = list(prompts.keys())
    raw_results = await asyncio.gather(*[_judge_one(prompts[k]) for k in keys])

    scores = {}
    for key, result in zip(keys, raw_results, strict=False):
        scores[key] = max(0.0, min(10.0, result.score))
        scores[f"{key}_reason"] = result.reason

    if "correctness" not in scores:
        scores["correctness"] = None
        scores["correctness_reason"] = "Эталонный ответ не задан"

    return scores


# ---------------------------------------------------------------------------
# Retriever метрики
# ---------------------------------------------------------------------------


def _extract_source_name(doc: Document) -> str:
    """Extract clean source name from document metadata."""
    filename = doc.metadata.get("filename", "")
    if filename:
        return filename
    source = doc.metadata.get("source", "")
    if source:
        return Path(source).name
    return "?"


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
            "retrieved_sources": [_extract_source_name(d) for d, _ in docs_with_scores],
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
        "retrieved_sources": [_extract_source_name(d) for d, _ in docs_with_scores],
    }


def compute_context_precision_recall(
    judge_llm: ChatOllama,
    question: str,
    answer: str,
    docs_with_scores: list[tuple[Document, float]],
) -> dict:
    """Compute context_precision and context_recall via LLM judge."""
    if not docs_with_scores:
        return {
            "context_precision": None,
            "context_precision_reason": "No documents retrieved",
            "context_recall": None,
            "context_recall_reason": "No documents retrieved",
        }

    client = _get_judge_client(settings.llm_model if settings.llm_provider == LLMProvider.OLLAMA else "")
    model = settings.llm_model if settings.llm_provider == LLMProvider.OLLAMA else settings.openrouter_model

    context = "\n\n---\n\n".join(d.page_content for d, _ in docs_with_scores)

    result = {"context_precision": None, "context_precision_reason": "", "context_recall": None}

    # Context precision
    try:
        prompt = CONTEXT_PRECISION_PROMPT.format(question=question, answer=answer, context=context)
        cp = _judge_with_structured_output(client, prompt, model)
        result["context_precision"] = max(0.0, min(10.0, cp.score))
        result["context_precision_reason"] = cp.reason
    except Exception as exc:
        logger.warning("Context precision judge failed: %s", exc)
        result["context_precision_reason"] = f"[Error: {exc}]"

    # Context recall
    try:
        prompt = CONTEXT_RECALL_PROMPT.format(question=question, answer=answer, context=context)
        cr = _judge_with_structured_output(client, prompt, model)
        result["context_recall"] = max(0.0, min(10.0, cr.score))
        result["context_recall_reason"] = cr.reason
    except Exception as exc:
        logger.warning("Context recall judge failed: %s", exc)
        result["context_recall_reason"] = f"[Error: {exc}]"

    return result


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

    cm = result.get("context_metrics", {})
    cp = cm.get("context_precision")
    cr = cm.get("context_recall")
    if cp is not None:
        logger.info("  Context Precision: %.1f/10  — %s", cp, cm.get("context_precision_reason", ""))
    if cr is not None:
        logger.info("  Context Recall:    %.1f/10  — %s", cr, cm.get("context_recall_reason", ""))

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

    cp_scores = [
        r["context_metrics"]["context_precision"]
        for r in results
        if r["context_metrics"].get("context_precision") is not None
    ]
    cr_scores = [
        r["context_metrics"]["context_recall"]
        for r in results
        if r["context_metrics"].get("context_recall") is not None
    ]

    return {
        "total_questions": len(results),
        "total_time_sec": round(sum(r["latency_sec"] for r in results), 1),
        "hit_rate": round(sum(hit_rates) / len(hit_rates), 3) if hit_rates else None,
        "avg_mrr": round(sum(mrrs) / len(mrrs), 3) if mrrs else None,
        "avg_faithfulness": round(sum(faiths) / len(faiths), 1) if faiths else None,
        "avg_relevancy": round(sum(rels) / len(rels), 1) if rels else None,
        "avg_correctness": round(sum(corrs) / len(corrs), 1) if corrs else None,
        "avg_similarity": round(sum(sims) / len(sims), 3) if sims else 0,
        "avg_context_precision": round(sum(cp_scores) / len(cp_scores), 1) if cp_scores else None,
        "avg_context_recall": round(sum(cr_scores) / len(cr_scores), 1) if cr_scores else None,
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
            "  Hit Rate:        %.1f/10  (%.0f%% вопросов нашли нужный источник)",
            m["hit_rate"] * 10,
            m["hit_rate"] * 100,
        )
        logger.info("  MRR:             %.3f  (1.0 = нужный чанк всегда первый)", m["avg_mrr"])
    logger.info("  Avg Similarity:  %.3f", m["avg_similarity"])

    logger.info("Context Quality:")
    if m["avg_context_precision"] is not None:
        logger.info(
            "  Context Precision: %.1f/10  (доля релевантных документов среди ретривированных)",
            m["avg_context_precision"],
        )
    if m["avg_context_recall"] is not None:
        logger.info(
            "  Context Recall:    %.1f/10  (доля нужной информации в контексте)", m["avg_context_recall"]
        )

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
    csv_header = (
        "id,question,faithfulness,relevancy,correctness,"
        "hit_rate,mrr,avg_sim,context_precision,context_recall,latency_sec"
    )
    rows = [csv_header]
    for r in results:
        gm = r["generator_metrics"]
        rm = r["retriever_metrics"]
        cm = r.get("context_metrics", {})
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
                    str(cm.get("context_precision", "") if cm.get("context_precision") is not None else ""),
                    str(cm.get("context_recall", "") if cm.get("context_recall") is not None else ""),
                    str(round(r["latency_sec"], 2)),
                ]
            )
        )
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    logger.info("Результаты сохранены:")
    logger.info("  JSON: %s", json_path)
    logger.info("  CSV:  %s", csv_path)

    # Append to history for trend tracking
    summary = compute_summary_metrics(results)

    config = {
        "top_k": settings.retriever_top_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "tei_embed_url": settings.tei_embed_url,
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
    merged_hashes = rrf_merge(
        [(h, v[0]) for h, v in dense_by_hash.items()] if dense_by_hash else [],
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
    logger.info("  provider  : %s", settings.llm_provider)
    logger.info("  rag model : %s", settings.llm_model)
    logger.info("  judge     : %s", judge_model)
    logger.info("  seed      : %s", seed if seed is not None else "none")
    logger.info("  n_runs    : %d", n_runs)
    logger.info("  qdrant    : %s", settings.qdrant_url)

    questions = load_questions(questions_path)

    fetch_k = settings.retriever_fetch_k

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
                judge_llm,
                question=q["question"],
                answer=answer,
                context=context_for_judge,
                expected_answer=q.get("expected_answer"),
            )

            context_metrics = compute_context_precision_recall(
                judge_llm,
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
    logger.info("  provider  : %s", settings.llm_provider)
    logger.info("  rag model : %s", settings.llm_model)
    logger.info("  judge     : %s", judge_model)
    logger.info("  seed      : %s", seed if seed is not None else "none")
    logger.info("  n_runs    : %d", n_runs)

    questions = load_questions(questions_path)
    fetch_k = settings.retriever_fetch_k

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
                    judge_llm,
                    question=q["question"],
                    answer=answer,
                    context=context_for_judge,
                    expected_answer=q.get("expected_answer"),
                )

                context_metrics = await asyncio.to_thread(
                    compute_context_precision_recall,
                    judge_llm,
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

    save_results(all_results, out_dir, model_name=settings.llm_model, run_id="all" if n_runs > 1 else None)
