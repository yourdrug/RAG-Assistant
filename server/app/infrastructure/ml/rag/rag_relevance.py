"""RAG relevance gate — Self-RAG relevance checking and citation filtering."""

import re

from infrastructure.ml.clients.llm_schemas import RelevanceCheck
from infrastructure.ml.rag.rag_formatting import format_docs

RELEVANCE_SYSTEM = (
    "Оцени, достаточно ли предоставленного контекста для ответа на вопрос пользователя.\n"
    "Если контекст достаточен — установи is_relevant=true.\n"
    "Если контекст недостаточен — установи is_relevant=false и кратко укажи причину.\n"
    "Не отвечай на сам вопрос — только оцени достаточность контекста."
)


async def check_relevance(ml_clients, question: str, docs: list) -> tuple[bool, str]:
    """Semantic check: check if the retrieved context is sufficient to answer the question.

    Uses instructor's ``max_retries=1`` (combined with auxiliary semaphore timeout
    to bound total hold time).
    """
    if not docs:
        return False, "Нет документов для проверки"

    import asyncio

    from config import settings
    from domain.value_objects.llm_provider import LLMProvider

    context = format_docs(docs, max_context_tokens=2000)
    prompt_text = f"Вопрос: {question}\n\nКонтекст из документов:\n{context}"

    client = ml_clients.instructor_client
    model = settings.llm_model if settings.llm_provider == LLMProvider.OLLAMA else settings.openrouter_model

    result = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": RELEVANCE_SYSTEM},
                {"role": "user", "content": prompt_text},
            ],
            response_model=RelevanceCheck,
            max_retries=1,
        ),
        timeout=settings.llm_auxiliary_timeout,
    )

    return result.is_relevant, result.reason


def filter_cited_sources(answer: str, sources: list[dict]) -> list[dict]:
    """Фильтрует источники, оставляя только те, на которые LLM действительно ссылалась."""
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    if not cited:
        return sources
    return [src for i, src in enumerate(sources, 1) if src.get("citation_id", i) in cited]


def filter_cited_documents(answer: str, docs: list) -> list:
    """Select prompt chunks by their original [N] labels before source aggregation."""
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    if not cited:
        return docs
    return [
        item
        for index, item in enumerate(docs, 1)
        if (item[0] if isinstance(item, tuple) else item).metadata.get("citation_id", index) in cited
    ]
