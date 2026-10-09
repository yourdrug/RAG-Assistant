"""Benchmark judge — LLM-as-judge scoring with structured output."""

import asyncio
import logging
import threading
import time
import json

import instructor
from instructor.core.exceptions import IncompleteOutputException
from openai import OpenAI, APIConnectionError, RateLimitError, InternalServerError
from pydantic import ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from typing import Any

from config import settings
from domain.services.rag_policy import build_system_prompt
from domain.value_objects.llm_provider import Breadth, LLMProvider
from langchain.schema import Document

from infrastructure.ml.clients.llm_schemas import JudgeScore
from infrastructure.ml.rag.rag_formatting import format_docs

from infrastructure.benchmark.judge_rubric import CORRECTNESS_CRITERIA, FAITHFULNESS_CRITERIA

logger = logging.getLogger("default")

JUDGE_MAX_RETRIES = 3
JUDGE_RETRY_DELAY = 5.0
JUDGE_OUTPUT_TOKENS = 2048
JUDGE_EXPANDED_OUTPUT_TOKENS = 8192

# ---------------------------------------------------------------------------
# Cached judge client (one connection pool per model, not per call)
# ---------------------------------------------------------------------------

_judge_client_cache: dict[str, Any] = {}
_judge_client_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANSWER_PROMPT_TEMPLATE = """\
{system_prompt}

Вопрос: {question}
"""

FAITHFULNESS_PROMPT = (
    """\
Ты — строгий эксперт по оценке качества ответов AI-ассистентов.

Контекст из документов:
{context}

Вопрос: {question}

Ответ ассистента: {answer}

Задача: """
    + FAITHFULNESS_CRITERIA
    + """

Ответь СТРОГО в формате JSON (только JSON, без пояснений):
{{"score": <число от 0 до 10>, "reason": "<не более 15 слов>"}}
"""
)

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
{{"score": <число от 0 до 10>, "reason": "<не более 15 слов>"}}
"""

CORRECTNESS_PROMPT = (
    """\
Ты — строгий эксперт по оценке качества ответов AI-ассистентов.

Вопрос: {question}

Эталонный ответ: {expected}

Ответ ассистента: {answer}

Задача: """
    + CORRECTNESS_CRITERIA
    + """

Ответь СТРОГО в формате JSON (только JSON, без пояснений):
{{"score": <число от 0 до 10>, "reason": "<не более 15 слов>"}}
"""
)

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
{{"score": <число от 0 до 10>, "reason": "<не более 15 слов>"}}
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
{{"score": <число от 0 до 10>, "reason": "<не более 15 слов>"}}
"""


# ---------------------------------------------------------------------------
# Judge client
# ---------------------------------------------------------------------------


def get_judge_model() -> str:
    """Resolve the judge model: explicit setting > fast model for OpenRouter > llm_model for Ollama."""
    if settings.benchmark_judge_model:
        return settings.benchmark_judge_model
    if settings.llm_provider == LLMProvider.OLLAMA:
        return settings.llm_model
    # OpenRouter: use fast/cheap model for judge calls
    return "meta-llama/llama-3.1-8b-instruct"


def get_judge_client(model: str):
    """Get or create a cached instructor client for the judge model.

    Thread-safe: first call creates the client, subsequent calls return the cached
    instance.  This avoids creating a new httpx connection pool per benchmark question.
    """
    cached = _judge_client_cache.get(model)
    if cached is not None:
        return cached
    with _judge_client_lock:
        cached = _judge_client_cache.get(model)
        if cached is not None:
            return cached
        # OpenRouter's Instructor adapter supports JSON, but rejects MD_JSON.
        # Both modes avoid requiring tool calling from the chosen model.
        raw = OpenAI(
            base_url=settings.openrouter_base_url
            if settings.llm_provider == LLMProvider.OPENROUTER
            else f"{settings.ollama_base_url}/v1",
            api_key=settings.openrouter_api_key
            if settings.llm_provider == LLMProvider.OPENROUTER
            else "ollama",
            timeout=settings.llm_auxiliary_timeout,
            max_retries=0,
        )
        mode = (
            instructor.Mode.JSON
            if settings.llm_provider == LLMProvider.OPENROUTER
            else instructor.Mode.MD_JSON
        )
        client = instructor.from_openai(raw, mode=mode)
        from infrastructure.benchmark.token_usage import record_judge_usage

        client.on("completion:response", record_judge_usage)
        _judge_client_cache[model] = client
        return client


def judge_with_structured_output(
    client,
    prompt: str,
    model: str,
) -> JudgeScore:
    """Call judge LLM with structured output via instructor.

    Uses two attempts for transient or validation failures; HTTP 400 is terminal.
    Truncation permits one additional call with a larger output budget and no retries.
    """
    initial_budget = settings.benchmark_judge_initial_tokens
    retry_budget = settings.benchmark_judge_retry_tokens
    for attempt, budget in enumerate((initial_budget, retry_budget)):
        try:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_model=JudgeScore,
                max_tokens=budget,
                timeout=settings.llm_auxiliary_timeout,
                max_retries=Retrying(
                    retry=retry_if_exception_type(
                        (
                            APIConnectionError,
                            RateLimitError,
                            InternalServerError,
                            ValidationError,
                            json.JSONDecodeError,
                        )
                    ),
                    stop=stop_after_attempt(2 if attempt == 0 else 1),
                    wait=wait_exponential(multiplier=1, max=3),
                ),
            )
        except IncompleteOutputException:
            if attempt == 1:
                raise
            logger.warning(
                "Judge output truncated for model %s at %d tokens; retrying with %d tokens",
                model,
                budget,
                retry_budget,
            )
    raise RuntimeError("Judge output budget exhausted")


# ---------------------------------------------------------------------------
# RAG answer generation
# ---------------------------------------------------------------------------


def get_rag_answer(llm, docs_with_scores: list[tuple[Document, float]], question: str) -> str:
    """Generate answer using the production system prompt and formatted context."""
    answer, _response = get_rag_answer_with_usage(llm, docs_with_scores, question)
    return answer


def get_rag_answer_with_usage(
    llm, docs_with_scores: list[tuple[Document, float]], question: str
) -> tuple[str, object]:
    """Generate answer and return (answer_text, raw_llm_response) for token usage extraction."""
    system_prompt = build_system_prompt(breadth=Breadth.NARROW.value)

    context = format_docs(docs_with_scores, max_context_tokens=4000)
    prompt_text = ANSWER_PROMPT_TEMPLATE.format(system_prompt=system_prompt, question=question)
    full_prompt = (
        f"{prompt_text}\n\nКонтекст из документов:\n<<DOCUMENT_CONTEXT>>\n{context}\n<<END_DOCUMENT_CONTEXT>>"
    )

    last_exc: Exception | None = None
    for attempt in range(1, JUDGE_MAX_RETRIES + 1):
        try:
            response = llm.invoke(full_prompt)
            content = response.content
            answer = content.strip() if isinstance(content, str) else str(content).strip()
            return answer, response
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
    if last_exc is None:
        raise RuntimeError("LLM judge failed with no exception recorded")
    raise last_exc


# ---------------------------------------------------------------------------
# Judge scoring
# ---------------------------------------------------------------------------


def judge_answer(
    question: str,
    answer: str,
    context: str,
    expected_answer: str | None = None,
    judge_llm=None,
) -> dict:
    """Run judge LLM synchronously with structured output (instructor only)."""
    model = get_judge_model()
    prompts = {
        "faithfulness": FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer),
        "relevancy": RELEVANCY_PROMPT.format(question=question, answer=answer),
    }
    if expected_answer:
        prompts["correctness"] = CORRECTNESS_PROMPT.format(
            question=question, expected=expected_answer, answer=answer
        )

    scores: dict[str, float | str | None] = {}
    for key, prompt in prompts.items():
        try:
            client = get_judge_client(model if settings.llm_provider == LLMProvider.OLLAMA else "")
            result = judge_with_structured_output(client, prompt, model)
            scores[key] = max(0.0, min(10.0, result.score))
            scores[f"{key}_reason"] = result.reason
        except Exception as exc:
            logger.warning("Judge structured output failed for %s: %s", key, exc)
            scores[key] = None
            scores[f"{key}_reason"] = f"[Ошибка вызова судьи: {exc}]"
            scores[f"{key}_error"] = str(exc)

    if "correctness" not in scores:
        scores["correctness"] = None
        scores["correctness_reason"] = "Эталонный ответ не задан"
    return scores


async def judge_answer_async(
    question: str,
    answer: str,
    context: str,
    expected_answer: str | None = None,
    judge_llm=None,
    *,
    judge_model: str | None = None,
    saved_metrics: dict | None = None,
    metric_callback=None,
) -> dict:
    """Judge answer quality with structured output (instructor + timeout)."""
    model = judge_model or get_judge_model()
    prompts = {
        "faithfulness": FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer),
        "relevancy": RELEVANCY_PROMPT.format(question=question, answer=answer),
    }
    if expected_answer:
        prompts["correctness"] = CORRECTNESS_PROMPT.format(
            question=question, expected=expected_answer, answer=answer
        )

    async def judge_one(prompt: str) -> JudgeScore:
        client = get_judge_client(model if settings.llm_provider == LLMProvider.OLLAMA else "")
        return await asyncio.wait_for(
            asyncio.to_thread(judge_with_structured_output, client, prompt, model),
            timeout=settings.llm_auxiliary_timeout * 3 + 3,
        )

    scores = dict(saved_metrics or {})
    for key, prompt in prompts.items():
        if scores.get(key) is not None:
            continue
        try:
            result = await judge_one(prompt)
            scores[key] = max(0.0, min(10.0, result.score))
            scores[f"{key}_reason"] = result.reason
            scores.pop(f"{key}_error", None)
        except Exception as exc:
            # Each metric is independent: provider failures are missing data,
            # never a numerical score, and must not discard sibling results.
            logger.warning("Judge failed for %s: %s", key, exc)
            scores[key] = None
            scores[f"{key}_error"] = str(exc)
            scores[f"{key}_reason"] = f"[Error: {exc}]"
        if metric_callback is not None:
            await metric_callback(dict(scores))

    if "correctness" not in scores:
        scores["correctness"] = None
        scores["correctness_reason"] = "Эталонный ответ не задан"

    return scores
