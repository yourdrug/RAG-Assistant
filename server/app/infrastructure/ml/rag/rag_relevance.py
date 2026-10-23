"""RAG relevance gate — Self-RAG relevance checking and citation filtering."""

import asyncio
import re

import config
from config import settings
from domain.rag.prompts import SUFFICIENCY_ASSESSMENT_SYSTEM
from domain.runtime.enums import LLMProvider
from infrastructure.ml.clients.client_lease import client_operation
from infrastructure.ml.clients.llm_schemas import RelevanceCheck, SufficiencyAssessment
from infrastructure.ml.rag.rag_formatting import format_docs
from infrastructure.resilience.circuit_breaker import get_breaker

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

    context = format_docs(docs, max_context_tokens=2000)
    prompt_text = f"Вопрос: {question}\n\nКонтекст из документов:\n{context}"

    client = ml_clients.instructor_client
    model = (
        config.settings.llm_model
        if config.settings.llm_provider == LLMProvider.OLLAMA
        else config.settings.openrouter_model
    )

    async with client_operation(client):
        try:
            result = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": RELEVANCE_SYSTEM},
                        {"role": "user", "content": prompt_text},
                    ],
                    max_tokens=config.settings.llm_num_predict_narrow,
                    response_model=RelevanceCheck,
                    max_retries=1,
                ),
                timeout=config.settings.llm_auxiliary_timeout,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"RAG relevance gate LLM request timed out "
                f"(provider={config.settings.llm_provider}, model={model}, "
                f"LLM_AUXILIARY_TIMEOUT={config.settings.llm_auxiliary_timeout}s)"
            ) from exc

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


async def assess_sufficiency(
    question: str, docs: list, ml_clients, llm_client=None, model: str = ""
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

        if settings.llm_provider == LLMProvider.OLLAMA:
            model = settings.llm_model
        else:
            model = settings.openrouter_model

    context = format_docs(docs, max_context_tokens=2000)
    user_msg = f"Вопрос: {question}\n\nКонтекст:\n{context}"

    breaker = get_breaker("llm_auxiliary")
    breaker.check_open()

    async with client_operation(llm_client), ml_clients.auxiliary_semaphore:
        try:
            result = await asyncio.wait_for(
                llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SUFFICIENCY_ASSESSMENT_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=settings.llm_num_predict_narrow,
                    response_model=SufficiencyAssessment,
                    max_retries=1,
                ),
                timeout=settings.llm_auxiliary_timeout,
            )
        except Exception:
            await breaker.report_failure()
            raise
        await breaker.report_success()
    return result
