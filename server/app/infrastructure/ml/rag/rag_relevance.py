"""RAG relevance gate — Self-RAG relevance checking and citation filtering."""

import re

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from infrastructure.ml.clients.llm_schemas import RelevanceCheck
from infrastructure.ml.rag.rag_formatting import format_docs

RELEVANCE_SYSTEM = (
    "Оцени, достаточно ли предоставленного контекста для ответа на вопрос пользователя.\n"
    "Если контекст достаточен — установи is_relevant=true.\n"
    "Если контекст недостаточен — установи is_relevant=false и кратко укажи причину.\n"
    "Не отвечай на сам вопрос — только оцени достаточность контекста."
)


def _get_rag_instructor_client():
    """Create instructor client for relevance checks (Ollama or OpenRouter)."""
    from infrastructure.llm.instructor_client import create_llm_instructor_client

    client, _model = create_llm_instructor_client()
    return client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
async def check_relevance(llm, question: str, docs: list) -> tuple[bool, str]:
    """Semantic check: check if the retrieved context is sufficient to answer the question."""
    if not docs:
        return False, "Нет документов для проверки"

    from config import settings
    from domain.value_objects.llm_provider import LLMProvider

    context = format_docs(docs, max_context_tokens=2000)
    prompt_text = f"Вопрос: {question}\n\nКонтекст из документов:\n{context}"

    client = _get_rag_instructor_client()
    model = settings.llm_model if settings.llm_provider == LLMProvider.OLLAMA else settings.openrouter_model

    import asyncio

    result = await asyncio.to_thread(
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": RELEVANCE_SYSTEM},
                {"role": "user", "content": prompt_text},
            ],
            response_model=RelevanceCheck,
            max_retries=3,
        )
    )

    return result.is_relevant, result.reason


def filter_cited_sources(answer: str, sources: list[dict]) -> list[dict]:
    """Фильтрует источники, оставляя только те, на которые LLM действительно ссылалась."""
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    if not cited:
        return sources
    filtered = [src for i, src in enumerate(sources, 1) if i in cited]
    return filtered if filtered else sources
