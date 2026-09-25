"""RAG prompts — query condensation, decomposition, summary, and prompt building."""

import asyncio
import logging

import httpx
from config import settings
from domain.services.rag_policy import build_system_prompt  # noqa: F401
from infrastructure.ml.clients.llm_schemas import DecompositionCheck
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

_RETRYABLE_ERRORS = (TimeoutError, ConnectionError, httpx.TransportError, OSError)

log = logging.getLogger("default")

# ---------------------------------------------------------------------------
# Query condensation
# ---------------------------------------------------------------------------

CONDENSE_SYSTEM = (
    "Учитывая историю диалога, перепиши следующий вопрос так, чтобы он был "
    "самодостаточным для поиска по документам. Сохрани смысл и язык вопроса. "
    "Если вопрос уже самодостаточен — верни его без изменений. "
    "Отвечай ТОЛЬКО переформулированным вопросом, без пояснений."
)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_SYSTEM),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10),
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    reraise=True,
)
async def condense_question(llm, question: str, history_messages: list, ml_clients=None) -> str:
    """Rewrite a follow-up question into a self-contained query using history context.

    If *ml_clients* is provided the auxiliary semaphore is acquired per-attempt
    (inside the retry loop) so that a retry storm does not hold the slot for
    the entire chain duration.
    """
    if not history_messages:
        return question

    chain = CONDENSE_PROMPT | llm

    async def _call():
        return await asyncio.wait_for(
            chain.ainvoke({"history": history_messages, "question": question}),
            timeout=settings.llm_auxiliary_timeout,
        )

    if ml_clients is not None:
        async with ml_clients.auxiliary_semaphore:
            result = await _call()
    else:
        result = await _call()

    condensed = result.content.strip()

    if not condensed or len(condensed) < 3:
        log.warning("Condensation returned empty/garbled output, using original question")
        return question

    len_ratio = len(condensed) / len(question) if len(question) > 0 else 1.0
    if len_ratio < 0.3 or len_ratio > 5.0:
        log.warning(
            "Condensation suspicious: len ratio %.2f, original=%r, condensed=%r",
            len_ratio,
            question,
            condensed,
        )

    log.info("Condensed query: %r -> %r (ratio=%.2f)", question, condensed, len_ratio)
    return condensed


# ---------------------------------------------------------------------------
# Query decomposition
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = (
    "Разбей составной вопрос на 2-4 независимых подвопроса. "
    "Каждый подвопрос должен быть самодостаточным для поиска по документам. "
    "Верни ТОЛЬКО список подвопросов, каждый на новой строке, без нумерации и маркеров."
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10),
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    reraise=True,
)
async def decompose_question(
    question: str,
    *,
    instructor_client=None,
    ml_clients=None,
) -> list[str]:
    """Split a compound question into independent sub-queries.

    Uses instructor with ``DecompositionCheck`` schema for structured output.
    Falls back to the original question if decomposition is not needed or fails.

    If *ml_clients* is provided the auxiliary semaphore is acquired per-attempt.
    """
    if instructor_client is None:
        log.warning("decompose_question called without instructor_client, returning original")
        return [question]

    from config import settings as _settings
    from domain.value_objects.llm_provider import LLMProvider

    model = (
        _settings.llm_model if _settings.llm_provider == LLMProvider.OLLAMA else _settings.openrouter_model
    )

    async def _call() -> DecompositionCheck:
        return await asyncio.wait_for(
            instructor_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": DECOMPOSE_SYSTEM},
                    {"role": "user", "content": question},
                ],
                response_model=DecompositionCheck,
                max_retries=2,
            ),
            timeout=_settings.llm_auxiliary_timeout,
        )

    if ml_clients is not None:
        async with ml_clients.auxiliary_semaphore:
            result = await _call()
    else:
        result = await _call()

    if not result.needs_decomposition or not result.sub_queries:
        log.info("Decomposition not needed for %r, using original question", question)
        return [question]

    sub_queries = [q.strip() for q in result.sub_queries if q.strip()]
    if len(sub_queries) < 2:
        log.warning("Decomposition returned %d sub-queries, using original question", len(sub_queries))
        return [question]

    log.info("Decomposed %r into %d sub-questions: %s", question, len(sub_queries), sub_queries)
    return sub_queries[:4]


# ---------------------------------------------------------------------------
# Rolling summary
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM = (
    "Составь краткое резюме диалога (3-5 предложений). "
    "Фиксируй ключевые факты, решения и контекст. "
    "Пиши на русском языке. Не начинай с «Резюме» — просто изложи суть."
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SUMMARY_SYSTEM),
        ("human", "{prompt}"),
    ]
)


async def update_rolling_summary(llm, existing_summary: str | None, new_turns: list[dict]) -> str:
    """Produce an updated rolling summary from existing summary + new dialog turns."""
    turns_text = "\n".join(
        f"{'Пользователь' if t['role'] == 'user' else 'Ассистент'}: {t['content'][:200]}" for t in new_turns
    )
    if existing_summary:
        prompt = f"Предыдущее резюме:\n{existing_summary}\n\nНовые сообщения:\n{turns_text}"
    else:
        prompt = f"Сообщения диалога:\n{turns_text}"

    chain = SUMMARY_PROMPT | llm
    result = await chain.ainvoke({"prompt": prompt})
    summary = result.content.strip()
    log.info("Rolling summary updated (%d chars)", len(summary))
    return summary


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(
    breadth: str = "narrow",
    summary: str | None = None,
    domain_addendum: str | None = None,
    enumerate_cases: bool = False,
) -> ChatPromptTemplate:
    system_text = build_system_prompt(
        breadth,
        domain_addendum=domain_addendum,
        enumerate_cases=enumerate_cases,
    )
    messages: list = [
        ("system", system_text),
    ]
    if summary:
        messages.append(("system", f"Резюме предыдущей части диалога:\n{summary}"))
    messages.append(MessagesPlaceholder(variable_name="history"))
    messages.append(("human", "{question}"))
    return ChatPromptTemplate.from_messages(messages)
