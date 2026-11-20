"""RAG relevance gate — Self-RAG relevance checking and citation filtering."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from config import LLMSection, request_section
from infrastructure.ml.clients.client_lease import client_operation
from infrastructure.ml.clients.llm_schemas import (
    RelevanceCheck,
    auxiliary_request_body,
    create_structured_response,
)
from infrastructure.ml.rag.benchmark_evidence import capture_gate
from infrastructure.ml.rag.rag_formatting import format_docs


if TYPE_CHECKING:
    from application.ports.ml_clients import MLClientPort

RELEVANCE_SYSTEM = (
    "Оцени, достаточно ли предоставленного контекста для ответа на вопрос пользователя.\n"
    "Если контекст достаточен — установи is_relevant=true.\n"
    "Если контекст недостаточен — установи is_relevant=false, кратко укажи причину и "
    "предложи в suggested_refinement уточнённый поисковый запрос, который найдёт недостающее; "
    "если уточнить нечего, оставь suggested_refinement пустым.\n"
    "Не отвечай на сам вопрос — только оцени достаточность контекста.\n"
    "Верни JSON-объект с полями is_relevant (boolean), reason (строка) и "
    "suggested_refinement (строка) на верхнем уровне, без дополнительных обёрток. "
    'Пример: {"is_relevant": false, "reason": "Нет данных", '
    '"suggested_refinement": "Уточнённый запрос"}.'
)


async def check_relevance(
    ml_clients: MLClientPort, question: str, docs: list[Any], *, context: str | None = None
) -> RelevanceCheck:
    """Semantic check: is the retrieved context sufficient, and if not, what to search next.

    One call replaces a separate sufficiency assessment after a rejection.

    Allows two Instructor attempts (one format correction), bounded together by
    the auxiliary timeout.
    """
    llm_settings = request_section(LLMSection)

    if not docs:
        return RelevanceCheck(is_relevant=False, reason="Нет документов для проверки")

    if context is None:
        context = format_docs(docs)

    prompt_text = f"Вопрос: {question}\n\nКонтекст из документов:\n{context}"

    client = ml_clients.instructor_client
    model = llm_settings.active_auxiliary_model

    async with client_operation(client):
        try:
            result = await asyncio.wait_for(
                create_structured_response(
                    client,
                    model=model,
                    messages=[
                        {"role": "system", "content": RELEVANCE_SYSTEM},
                        {"role": "user", "content": prompt_text},
                    ],
                    max_tokens=llm_settings.llm_num_predict_narrow,
                    response_model=RelevanceCheck,
                    max_retries=2,
                    extra_body=auxiliary_request_body(llm_settings),
                ),
                timeout=llm_settings.llm_auxiliary_timeout,
            )
        except TimeoutError as exc:
            message = (
                f"RAG relevance gate LLM request timed out "
                f"(model={model}, "
                f"LLM_AUXILIARY_TIMEOUT={llm_settings.llm_auxiliary_timeout}s)"
            )
            raise TimeoutError(message) from exc

    capture_gate("relevance", question, context, result.model_dump())
    return result


def filter_cited_sources(answer: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Фильтрует источники, оставляя только те, на которые LLM действительно ссылалась."""
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}

    if not cited:
        return sources

    return [src for i, src in enumerate(sources, 1) if src.get("citation_id", i) in cited]


def filter_cited_documents(answer: str, docs: list[Any]) -> list[Any]:
    """Select prompt chunks by their original [N] labels before source aggregation."""
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}

    if not cited:
        return docs

    return [
        item
        for index, item in enumerate(docs, 1)
        if (item[0] if isinstance(item, tuple) else item).metadata.get("citation_id", index) in cited
    ]
