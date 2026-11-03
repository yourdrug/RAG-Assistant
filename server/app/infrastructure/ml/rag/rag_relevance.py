"""RAG relevance gate — Self-RAG relevance checking and citation filtering."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from application.runtime.enums import LLMProvider
from config import get_request_settings
from infrastructure.ml.clients.client_lease import client_operation
from infrastructure.ml.clients.llm_schemas import (
    RelevanceCheck,
    SufficiencyAssessment,
    create_structured_response,
)
from infrastructure.ml.rag.prompts import SUFFICIENCY_ASSESSMENT_SYSTEM
from infrastructure.ml.rag.rag_formatting import format_docs
from infrastructure.resilience.circuit_breaker import get_breaker


if TYPE_CHECKING:
    from application.ports.ml_clients import MLClientPort

RELEVANCE_SYSTEM = (
    "Оцени, достаточно ли предоставленного контекста для ответа на вопрос пользователя.\n"
    "Если контекст достаточен — установи is_relevant=true.\n"
    "Если контекст недостаточен — установи is_relevant=false и кратко укажи причину.\n"
    "Не отвечай на сам вопрос — только оцени достаточность контекста."
)


async def check_relevance(ml_clients: MLClientPort, question: str, docs: list[Any]) -> tuple[bool, str]:
    """Semantic check: check if the retrieved context is sufficient to answer the question.

    Uses instructor's ``max_retries=1`` (combined with auxiliary semaphore timeout
    to bound total hold time).
    """
    if not docs:
        return False, "Нет документов для проверки"

    context = format_docs(docs, max_context_tokens=2000)
    prompt_text = f"Вопрос: {question}\n\nКонтекст из документов:\n{context}"

    client = ml_clients.instructor_client
    model = (
        get_request_settings().llm_model
        if get_request_settings().llm_provider == LLMProvider.OLLAMA
        else get_request_settings().openrouter_model
    )

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
                    max_tokens=get_request_settings().llm_num_predict_narrow,
                    response_model=RelevanceCheck,
                    max_retries=1,
                ),
                timeout=get_request_settings().llm_auxiliary_timeout,
            )
        except TimeoutError as exc:
            message = (
                f"RAG relevance gate LLM request timed out "
                f"(provider={get_request_settings().llm_provider}, model={model}, "
                f"LLM_AUXILIARY_TIMEOUT={get_request_settings().llm_auxiliary_timeout}s)"
            )
            raise TimeoutError(message) from exc

    return result.is_relevant, result.reason


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


async def assess_sufficiency(
    question: str,
    docs: list[Any],
    ml_clients: MLClientPort,
    llm_client: object | None = None,
    model: str = "",
) -> SufficiencyAssessment:
    """Self-RAG: assess whether retrieved context is sufficient to answer.

    Uses instructor's ``max_retries=1`` (combined with auxiliary semaphore timeout
    to bound total hold time).
    """
    if not docs:
        return SufficiencyAssessment(
            is_sufficient=False,
            reasoning="No documents retrieved",
            suggested_refinement=question,
        )

    if llm_client is None:
        llm_client = ml_clients.instructor_client

        if get_request_settings().llm_provider == LLMProvider.OLLAMA:
            model = get_request_settings().llm_model
        else:
            model = get_request_settings().openrouter_model

    context = format_docs(docs, max_context_tokens=2000)
    user_msg = f"Вопрос: {question}\n\nКонтекст:\n{context}"

    breaker = get_breaker("llm_auxiliary")
    breaker.check_open()

    async with client_operation(llm_client), ml_clients.auxiliary_semaphore:
        try:
            result = await asyncio.wait_for(
                create_structured_response(
                    llm_client,
                    model=model,
                    messages=[
                        {"role": "system", "content": SUFFICIENCY_ASSESSMENT_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=get_request_settings().llm_num_predict_narrow,
                    response_model=SufficiencyAssessment,
                    max_retries=1,
                ),
                timeout=get_request_settings().llm_auxiliary_timeout,
            )
        except Exception:
            await breaker.report_failure()
            raise
        await breaker.report_success()
    return result
