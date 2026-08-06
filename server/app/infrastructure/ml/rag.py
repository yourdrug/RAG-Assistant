"""
infrastructure/ml/rag.py — RAG logic using LangChain: prompts, reranking, formatting, source extraction.
Moved from domain/rag.py to keep domain free of LangChain dependencies.
"""

import asyncio
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
    """Classify question as 'narrow' or 'broad' based on heuristics."""
    q = question.lower()

    # Narrow exceptions: even if a broad pattern matches, these stay narrow
    narrow_overrides = [
        r"как\w*\s+(убедиться|проверить|узнать|найти|получить|скачать|открыть)",
        r"где\s+(найти|скачать|посмотреть|открыть)",
        r"что\s+(такое|означает|является)",
        r"какой\s+(пароль|срок|размер|номер|формат|статус)",
        r"каки[ех]\s+исключени",
        r"каки[ех]\s+особы",
        r"каки[ех]\s+альтернатив",
        r"почему\s+",
        r"можно\s+ли\s+",
    ]
    if any(re.search(p, q) for p in narrow_overrides):
        return "narrow"

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
        r"какие\s+\w+\s+нужн",
    ]
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
3. Отвечай на том же языке, на котором задан вопрос.
4. Используй точные термины и сокращения из документов. НЕ заменяй сокращения (например: ЭТТН, ЭТН, ИМН — пиши как в источнике). Не подменяй их другими аббревиатурами.
5. Указывай номера страниц (например: "см. стр. 3, 7"), если они есть в контексте.
6. Если в контексте есть частичная информация — укажи только то, что есть, и скажи чего не хватает.
7. Если в контексте есть ссылки на изображения [image: ...] — ОБЯЗАТЕЛЬНО включай их в ответ. Не удаляй и не игнорируй.
8. НЕ ОТКЛОНЯЙСЯ от темы вопроса. Не давай общую информацию по теме, если она не запрашивалась. Отвечай строго на заданный вопрос — ни больше, ни меньше.

Контекст из документов:
{context}
"""


def build_prompt(breadth: str = "narrow") -> ChatPromptTemplate:
    if breadth == "broad":
        rule3 = (
            "3. Отвечай РАЗВЁРНУТО по структуре:\n"
            "   - Начни с краткого прямого ответа (1 предложение)\n"
            "   - Затем раскрой тему по подпунктам: 1-2 предложения с деталями из контекста\n"
            "   - Заверши нюансами/исключениями, если они есть в контексте\n"
            "   Не пересказывай весь документ — освещай аспекты заданного вопроса."
        )
    else:
        rule3 = (
            "3. Отвечай КРАТКО: 1-3 предложения. Только прямой ответ на вопрос.\n"
            "   Не добавляй контекст, не относящийся напрямую к вопросу.\n"
            "   Не перечисляй всё из документа — отвечай конкретно на то, что спрашивают.\n"
            "   Если вопрос — о конкретном факте (дата, цифра, название), назови только его.\n"
            "   Не расширяй тему: если спросили про X — не рассказывайте про Y, даже если он связан."
        )

    system_text = SYSTEM_PROMPT.replace(
        "3. Отвечай на том же языке, на котором задан вопрос.",
        rule3,
    )

    system_text = system_text.replace(
        "4. Используй точные термины",
        "4. Используй точные термины",
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


async def rerank_documents(
        question: str,
        docs: list,
        top_n: int,
        reranker=None,
        min_score: float | None = None,
        score_gap_ratio: float | None = None,
) -> list[tuple]:
    """
    Переранжирует кандидатов кросс-энкодером и возвращает top_n лучших
    как список пар (doc, score).

    Фильтрация:
      - min_score: отбросить чанки с score < min_score (абсолютный порог)
      - score_gap_ratio: отбросить чанки, чей score ниже top-1 более чем в N раз
        (например score_gap_ratio=0.1 означает «оставить всё ≥ 10% от лучшего»)

    reranker — объект с методом .predict(pairs).
    """
    if not docs:
        return []

    pairs = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        filename = doc.metadata.get("filename", "")
        doc_name = filename or Path(source).name if source else ""
        content_with_prefix = f"[{doc_name}] {doc.page_content}" if doc_name else doc.page_content
        pairs.append((question, content_with_prefix))

    scores = await asyncio.to_thread(reranker.predict, pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_n]

    if min_score is not None:
        ranked = [(d, s) for d, s in ranked if s >= min_score]

    if score_gap_ratio is not None and ranked:
        top_score = ranked[0][1]
        cutoff = top_score * score_gap_ratio
        ranked = [(d, s) for d, s in ranked if s >= cutoff]

    return ranked


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

    Принимает list[Document] или list[tuple[Document, float]] (после rerank_documents).
    Respects context budget: truncates docs list if total estimated tokens exceed limit.
    qwen2.5:14b supports ~32k context, but we reserve space for system prompt + history + response.
    """
    parts = []
    total_chars = 0
    max_chars = max_context_tokens * CHARS_PER_TOKEN

    for i, item in enumerate(docs, 1):
        doc = item[0] if isinstance(item, tuple) else item
        source = doc.metadata.get("source", "unknown")
        source_name = _clean_source_name(source)
        page = doc.metadata.get("page")
        page_start = doc.metadata.get("page_start")
        page_end = doc.metadata.get("page_end")
        doc_date = doc.metadata.get("doc_date")
        header = f"[{i}] {source_name}"

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


def _clean_source_name(source: str) -> str:
    """Extract clean filename from full path, strip directory."""
    from pathlib import Path

    return Path(source).name if source else "unknown"


def extract_sources(docs, min_score: float | None = None) -> list[dict]:
    """Извлекает метаданные источников для сохранения в БД.

    Принимает list[Document] или list[tuple[Document, float]] (после rerank_documents).
    Если переданы пары (doc, score), в каждый источник добавляется max_score
    и список сортируется по убыванию max_score (самый релевантный — первый).

    min_score: если задан, источники с max_score < min_score отбрасываются.
    """
    pages_by_source: dict[str, set[str]] = {}
    scores_by_source: dict[str, float] = {}
    for item in docs:
        doc = item[0] if isinstance(item, tuple) else item
        score = item[1] if isinstance(item, tuple) else None
        src = doc.metadata.get("source", "unknown")
        clean_name = _clean_source_name(src)
        page = doc.metadata.get("page")
        page_start = doc.metadata.get("page_start")
        page_end = doc.metadata.get("page_end")
        pages_list = doc.metadata.get("pages")
        if clean_name not in pages_by_source:
            pages_by_source[clean_name] = set()
        if pages_list:
            pages_by_source[clean_name].update(pages_list)
        elif page_start is not None and page_end is not None:
            pages_by_source[clean_name].update(range(page_start, page_end + 1))
        elif page is not None:
            pages_by_source[clean_name].add(page)
        if score is not None:
            prev = scores_by_source.get(clean_name, float("-inf"))
            if score > prev:
                scores_by_source[clean_name] = score

    sources = []
    for src, pages in pages_by_source.items():
        sorted_pages = sorted(pages) if pages else []
        entry = {
            "source": src,
            "pages": sorted_pages,
        }
        if scores_by_source:
            entry["max_score"] = round(float(scores_by_source.get(src, 0.0)), 4)
        sources.append(entry)

    if scores_by_source:
        sources.sort(key=lambda s: s.get("max_score", 0.0), reverse=True)

    if min_score is not None and sources and scores_by_source:
        sources = [s for s in sources if s.get("max_score", 0.0) >= min_score]

    return sources


def filter_cited_sources(answer: str, sources: list[dict]) -> list[dict]:
    """Фильтрует источники, оставляя только те, на которые LLM действительно ссылалась.

    Ищет паттерны [N] в ответе и возвращает источники с соответствующими индексами (1-based).
    Если в ответе нет ни одной ссылки или ни один индекс не совпадает — возвращаем все источники.
    """
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    if not cited:
        return sources
    filtered = [src for i, src in enumerate(sources, 1) if i in cited]
    return filtered if filtered else sources
