"""LangChain-based RAG helpers -- prompts, reranking, formatting, and source extraction.

Pure policy functions (breadth classification, prompt building) live in
``domain/services/rag_policy.py``; this module handles LangChain-specific
construction (chain assembly, document formatting, citation extraction).
"""

import asyncio
import logging
import re
from pathlib import Path

from domain.services.rag_policy import build_system_prompt, classify_question_breadth  # noqa: F401
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

log = logging.getLogger("default")

# Approximate tokens per character for Russian text (~4 chars per token)
CHARS_PER_TOKEN = 4


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


def build_prompt(breadth: str = "narrow", has_legal_context: bool = False) -> ChatPromptTemplate:
    system_text = build_system_prompt(breadth, has_legal_context=has_legal_context)
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
        article_number = doc.metadata.get("article_number")
        header = f"[{i}] {source_name}"

        if doc_date:
            header += f" от {doc_date}"

        if article_number:
            header += f", ст. {article_number}"
        elif page_start is not None and page_end is not None and page_start != page_end:
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
    articles_by_source: dict[str, list[str]] = {}
    for item in docs:
        doc = item[0] if isinstance(item, tuple) else item
        score = item[1] if isinstance(item, tuple) else None
        src = doc.metadata.get("source", "unknown")
        clean_name = _clean_source_name(src)
        page = doc.metadata.get("page")
        page_start = doc.metadata.get("page_start")
        page_end = doc.metadata.get("page_end")
        pages_list = doc.metadata.get("pages")
        article_number = doc.metadata.get("article_number")
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
        if article_number:
            articles_by_source.setdefault(clean_name, [])
            if article_number not in articles_by_source[clean_name]:
                articles_by_source[clean_name].append(article_number)

    sources = []
    for src, pages in pages_by_source.items():
        sorted_pages = sorted(pages) if pages else []
        entry = {
            "source": src,
            "pages": sorted_pages,
        }
        if src in articles_by_source:
            entry["articles"] = articles_by_source[src]
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
