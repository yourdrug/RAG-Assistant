"""
infrastructure/ml/rag.py — RAG logic using LangChain: prompts, reranking, formatting, source extraction.
Moved from domain/rag.py to keep domain free of LangChain dependencies.
"""

import logging
import re
from pathlib import Path

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

log = logging.getLogger("default")

# Approximate tokens per character for Russian text (~4 chars per token)
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Question breadth classification
# ---------------------------------------------------------------------------


def classify_question_breadth(question: str) -> str:
    """Classify question as 'narrow' or 'broad' based on simple heuristics."""
    broad_patterns = [
        r"подробно",
        r"объясни\s+вс[ёе]",
        r"расскажи\s+про",
        r"как\w*\s+(работает|устроено|происходит|проводится)",
        r"порядок\s+(получения|выдачи|оформления|получить)",
        r"система\s+\w+",
        r"вс[ёе]\s+про\b",
        r"полностью",
        r"детальн",
        r"максимальн",
        r"каки[ех]\s+(условия|требования|правила|нормы|критерии)",
        r"перечисл\w*",
        r"что\s+входит",
        r"что\s+включ\w+",
        r"список\s+\w+",
        r"какие\s+\w+\s+существуют",
        r"какие\s+\w+\s+нужн",
    ]
    q = question.lower()
    return "broad" if any(re.search(p, q) for p in broad_patterns) else "narrow"


# ---------------------------------------------------------------------------
# Query condensation prompt
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


async def condense_question(llm, question: str, history_messages: list) -> str:
    """Rewrite a follow-up question into a self-contained query using history context.

    If there is no history or only one turn, returns the original question unchanged.
    Logs original → condensed pairs for debugging and quality monitoring.
    """
    if not history_messages:
        return question

    chain = CONDENSE_PROMPT | llm
    result = await chain.ainvoke({"history": history_messages, "question": question})
    condensed = result.content.strip()

    if not condensed or len(condensed) < 3:
        log.warning("Condensation returned empty/garbled output, using original question")
        return question

    # Validate: condensed should not be drastically shorter or longer
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


def build_prompt(breadth: str = "narrow") -> ChatPromptTemplate:
    brevity_rule = (
        "3. Отвечай РАЗВЁРНУТО: раскрывай тему полностью, с примерами и деталями."
        if breadth == "broad"
        else "3. Отвечай КРАТКО: 1-3 предложения. Не пересказывай весь документ — отвечай конкретно на заданный вопрос."
    )
    system_text = SYSTEM_PROMPT.replace(
        "3. Отвечай КРАТКО: 1-3 предложения. Не пересказывай весь документ — отвечай конкретно на заданный вопрос.",
        brevity_rule,
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
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

    Adds document name prefix to reranker input for better context.
    """
    if not docs:
        return docs

    pairs = []
    for doc in docs:
        # Add source filename as prefix for reranker context
        source = doc.metadata.get("source", "")
        filename = doc.metadata.get("filename", "")
        doc_name = filename or Path(source).name if source else ""
        content_with_prefix = f"[{doc_name}] {doc.page_content}" if doc_name else doc.page_content
        pairs.append((question, content_with_prefix))

    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _score in ranked[:top_n]]


def deduplicate_docs(docs: list) -> list:
    """Remove near-duplicate chunks by content_hash to improve context diversity.

    When overlap is large or documents have similar formulations,
    the retriever may return 2-3 nearly identical chunks.
    This function keeps only unique chunks by their content hash.
    """
    from infrastructure.ml.hybrid import content_hash

    seen_hashes = set()
    unique_docs = []
    for doc in docs:
        h = content_hash(doc.page_content)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)
    return unique_docs


def format_docs(docs, max_context_tokens: int = 6000) -> str:
    """Форматирует найденные чанки в строку для промпта.

    Respects context budget: truncates docs list if total estimated tokens exceed limit.
    qwen2.5:14b supports ~32k context, but we reserve space for system prompt + history + response.
    """
    parts = []
    total_chars = 0
    max_chars = max_context_tokens * CHARS_PER_TOKEN

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        page_start = doc.metadata.get("page_start")
        page_end = doc.metadata.get("page_end")
        doc_date = doc.metadata.get("doc_date")
        header = f"[{i}] {source}"

        if doc_date:
            header += f" от {doc_date}"

        if page_start is not None and page_end is not None and page_start != page_end:
            header += f" (стр. {page_start}-{page_end})"

        elif page is not None:
            header += f" (стр. {page})"

        content = doc.page_content
        part_text = f"{header}\n{content}"
        part_chars = len(part_text)

        # Check if adding this doc would exceed budget
        separator_len = 6  # "\n\n---\n\n"
        if total_chars + part_chars > max_chars and parts:
            log.warning(
                "Context budget reached: %d/%d tokens, stopping at %d docs",
                total_chars // CHARS_PER_TOKEN,
                max_context_tokens,
                len(parts),
            )
            break

        parts.append(part_text)
        total_chars += part_chars + separator_len

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
        page_start = doc.metadata.get("page_start")
        page_end = doc.metadata.get("page_end")
        pages_list = doc.metadata.get("pages")
        if src not in pages_by_source:
            pages_by_source[src] = set()
        if pages_list:
            pages_by_source[src].update(pages_list)
        elif page_start is not None and page_end is not None:
            pages_by_source[src].update(range(page_start, page_end + 1))
        elif page is not None:
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
