"""Benchmark judge — LLM-as-judge scoring with structured output."""

import asyncio
import logging
import time

from config import settings
from domain.services.rag_policy import build_system_prompt
from domain.value_objects.llm_provider import Breadth, LLMProvider
from langchain.schema import Document
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from infrastructure.ml.clients.llm_schemas import JudgeScore
from infrastructure.ml.rag import format_docs

logger = logging.getLogger("default")

JUDGE_MAX_RETRIES = 3
JUDGE_RETRY_DELAY = 5.0

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANSWER_PROMPT_TEMPLATE = """\
{system_prompt}

Вопрос: {question}
"""

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


# ---------------------------------------------------------------------------
# Judge client
# ---------------------------------------------------------------------------


def _get_judge_client(model: str):
    """Create an instructor-wrapped client for the judge model."""
    from infrastructure.llm.instructor_client import create_llm_instructor_client

    client, _resolved = create_llm_instructor_client(model=model)
    return client


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


# ---------------------------------------------------------------------------
# RAG answer generation
# ---------------------------------------------------------------------------


def get_rag_answer(llm, docs_with_scores: list[tuple[Document, float]], question: str) -> str:
    """Generate answer using the production system prompt and formatted context."""
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
            return content.strip() if isinstance(content, str) else str(content).strip()
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

    scores: dict[str, float | str | None] = {}
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
    question: str,
    answer: str,
    context: str,
    expected_answer: str | None = None,
    judge_llm=None,
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

        def _call() -> JudgeScore:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_model=JudgeScore,
                max_retries=3,
            )

        for attempt in range(1, JUDGE_MAX_RETRIES + 1):
            try:
                result = await _asyncio.to_thread(_call)
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

    scores: dict[str, float | str | None] = {}
    for key, result in zip(keys, raw_results, strict=False):
        scores[key] = max(0.0, min(10.0, result.score))
        scores[f"{key}_reason"] = result.reason

    if "correctness" not in scores:
        scores["correctness"] = None
        scores["correctness_reason"] = "Эталонный ответ не задан"

    return scores
