"""
infrastructure/ml/rag.py — RAG logic using LangChain: prompts, reranking, formatting, source extraction.
Moved from domain/rag.py to keep domain free of LangChain dependencies.
"""

import logging

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

log = logging.getLogger("default")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ты — корпоративный ассистент. Строгие правила:

1. Отвечай ТОЛЬКО на основе предоставленного контекста. Контекст — единственный источник правды.
2. Если ответа нет в контексте — ответь ровно: "Информация не найдена в документах." Не придумывай и не додумывай.
3. Отвечай КРАТКО: 1-3 предложения. Не пересказывай весь документ — отвечай конкретно на заданный вопрос.
4. Отвечай на том же языке, на котором задан вопрос.
5. Указывай номера страниц (например: "см. стр. 3, 7"), если они есть в контексте.
6. Если в контексте есть частичная информация — укажи только то, что есть, и скажи чего не хватает.

Контекст из документов:
{context}
"""


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def rerank_documents(question: str, docs: list, top_n: int, reranker=None) -> list:
    """
    Переранжирует кандидатов кросс-энкодером и возвращает top_n лучших.
    reranker — объект с методом .predict(pairs).
    """
    if not docs:
        return docs

    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _score in ranked[:top_n]]


def format_docs(docs) -> str:
    """Форматирует найденные чанки в строку для промпта."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        header = f"[{i}] {source}"
        if page is not None:
            header += f" (стр. {page})"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def history_to_messages(history: list[dict]):
    """Конвертирует историю из БД в LangChain-сообщения."""
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


def extract_sources(docs) -> list[dict]:
    """Извлекает метаданные источников для сохранения в БД."""
    pages_by_source: dict[str, set[int]] = {}
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        if src not in pages_by_source:
            pages_by_source[src] = set()
        if page is not None:
            pages_by_source[src].add(page)

    sources = []
    for src, pages in pages_by_source.items():
        sorted_pages = sorted(pages) if pages else []
        sources.append(
            {
                "source": src,
                "pages": sorted_pages,
            }
        )
    return sources
