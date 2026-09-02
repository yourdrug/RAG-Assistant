"""LangChain-based RAG helpers -- prompts, reranking, formatting, and source extraction.

Pure policy functions (breadth classification, prompt building) live in
``domain/services/rag_policy.py``; this module handles LangChain-specific
construction (chain assembly, document formatting, citation extraction).
"""

import logging
import re
from pathlib import Path

from domain.services.rag_policy import (  # noqa: F401
    build_system_prompt,
    classify_question_breadth,
    needs_decomposition,
)
from domain.value_objects.message_role import MessageRole
from domain.value_objects.page_content_type import PageContentType
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from infrastructure.ml.hybrid import content_hash
from infrastructure.ml.llm_schemas import RelevanceCheck

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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
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
# Query decomposition for compound questions
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = (
    "Разбей составной вопрос на 2-4 независимых подвопроса. "
    "Каждый подвопрос должен быть самодостаточным для поиска по документам. "
    "Верни ТОЛЬКО список подвопросов, каждый на новой строке, без нумерации и маркеров."
)

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", DECOMPOSE_SYSTEM),
        ("human", "{question}"),
    ]
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
async def decompose_question(llm, question: str) -> list[str]:
    """Split a compound question into independent sub-queries.

    Returns a list of 2-4 sub-questions.  If decomposition fails or produces
    a single line, returns the original question as the only element.
    """
    chain = DECOMPOSE_PROMPT | llm
    result = await chain.ainvoke({"question": question})
    lines = [line.strip() for line in result.content.strip().split("\n") if line.strip()]

    if len(lines) < 2:
        log.warning("Decomposition returned %d lines, using original question", len(lines))
        return [question]

    log.info("Decomposed %r into %d sub-questions: %s", question, len(lines), lines)
    return lines[:4]


# ---------------------------------------------------------------------------
# Rolling summary for long dialogs
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
    """Produce an updated rolling summary from existing summary + new dialog turns.

    Called fire-and-forget after saving the assistant response.  New turns are
    truncated to 200 chars each to keep the summary prompt compact.
    """
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
# Prompt
# ---------------------------------------------------------------------------


def build_prompt(
    breadth: str = "narrow",
    has_legal_context: bool = False,
    summary: str | None = None,
) -> ChatPromptTemplate:
    system_text = build_system_prompt(breadth, has_legal_context=has_legal_context)
    messages: list = [
        ("system", system_text),
    ]
    if summary:
        messages.append(("system", f"Резюме предыдущей части диалога:\n{summary}"))
    messages.append(MessagesPlaceholder(variable_name="history"))
    messages.append(("human", "{question}"))
    return ChatPromptTemplate.from_messages(messages)


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
    """Переранжировать кандидатов кросс-энкодером и вернуть top_n лучших.

    Возвращает список пар (doc, score).  Фильтрация:
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

    scores = reranker.predict(pairs)
    if hasattr(scores, "__await__"):
        scores = await scores

    ranked = sorted(zip(docs, scores, strict=False), key=lambda x: x[1], reverse=True)[:top_n]

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
    parts: list[str] = []
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

        content_type = doc.metadata.get("content_type")
        if content_type == PageContentType.TABLE.value:
            header += " (таблица)"

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
    messages: list[HumanMessage | AIMessage] = []
    for msg in history:
        if msg["role"] == MessageRole.USER.value:
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


def _clean_source_name(source: str) -> str:
    """Extract clean filename from full path, strip directory."""
    return Path(source).name if source else "unknown"


def _collect_source_metadata(
    doc, score: float | None
) -> tuple[str, set[str], float | None, str | None, bool, bool, str | None, int | None, str | None]:
    """Extract pages, score, articles, flags, document_id, content_hash from a single doc.

    Returns (clean_name, pages_set, score, article_number, is_edited,
    is_manual, edited_at, document_id, content_hash).
    """
    src = doc.metadata.get("source", "unknown")
    clean_name = _clean_source_name(src)
    page = doc.metadata.get("page")
    page_start = doc.metadata.get("page_start")
    page_end = doc.metadata.get("page_end")
    pages_list = doc.metadata.get("pages")
    article_number = doc.metadata.get("article_number")
    is_edited = doc.metadata.get("edited", False)
    is_manual = doc.metadata.get("manual", False)
    edited_at = doc.metadata.get("edited_at")
    document_id = doc.metadata.get("document_id")
    content_hash = doc.metadata.get("content_hash")

    pages_set: set[str] = set()
    if pages_list:
        pages_set.update(pages_list)
    elif page_start is not None and page_end is not None:
        pages_set.update(str(p) for p in range(page_start, page_end + 1))
    elif page is not None:
        pages_set.add(page)

    return (
        clean_name,
        pages_set,
        score,
        article_number,
        is_edited,
        is_manual,
        edited_at,
        document_id,
        content_hash,
    )


def _build_source_entry(
    src: str,
    pages: set[str],
    articles_by_source: dict[str, list[str]],
    scores_by_source: dict[str, float],
    edited_by_source: dict[str, bool],
    manual_by_source: dict[str, bool],
    edited_at_by_source: dict[str, str | None],
    document_ids_by_source: dict[str, int | None],
    content_hashes_by_source: dict[str, list[str]],
) -> dict:
    """Build the entry dict for a single source."""
    sorted_pages = sorted(pages) if pages else []
    entry: dict = {
        "source": src,
        "pages": sorted_pages,
    }
    if src in document_ids_by_source:
        entry["document_id"] = document_ids_by_source[src]
    if src in content_hashes_by_source and content_hashes_by_source[src]:
        entry["content_hashes"] = content_hashes_by_source[src]
    if src in articles_by_source:
        entry["articles"] = articles_by_source[src]
    if scores_by_source:
        entry["max_score"] = round(float(scores_by_source.get(src, 0.0)), 4)
    if edited_by_source.get(src):
        entry["edited"] = True
    if manual_by_source.get(src):
        entry["manual"] = True
    if edited_at_by_source.get(src):
        entry["edited_at"] = edited_at_by_source[src]
    return entry


def _update_article_list(
    articles_by_source: dict[str, list[str]], clean_name: str, article_number: str
) -> None:
    articles_by_source.setdefault(clean_name, [])
    if article_number not in articles_by_source[clean_name]:
        articles_by_source[clean_name].append(article_number)


def _update_edited_at(edited_at_by_source: dict[str, str | None], clean_name: str, edited_at: str) -> None:
    prev_edited_at = edited_at_by_source.get(clean_name)
    if prev_edited_at is None or edited_at > prev_edited_at:
        edited_at_by_source[clean_name] = edited_at


def _aggregate_source_metadata(  # noqa: C901
    item,
    pages_by_source: dict[str, set[str]],
    scores_by_source: dict[str, float],
    articles_by_source: dict[str, list[str]],
    edited_by_source: dict[str, bool],
    manual_by_source: dict[str, bool],
    edited_at_by_source: dict[str, str | None],
    document_ids_by_source: dict[str, int | None],
    content_hashes_by_source: dict[str, list[str]],
) -> None:
    doc = item[0] if isinstance(item, tuple) else item
    score = item[1] if isinstance(item, tuple) else None
    clean_name, pages_set, doc_score, article_number, is_edited, is_manual, edited_at, document_id, ch = (
        _collect_source_metadata(doc, score)
    )

    if clean_name not in pages_by_source:
        pages_by_source[clean_name] = set()
    if pages_set:
        pages_by_source[clean_name].update(pages_set)
    if doc_score is not None:
        prev = scores_by_source.get(clean_name, float("-inf"))
        if doc_score > prev:
            scores_by_source[clean_name] = doc_score
    if article_number:
        _update_article_list(articles_by_source, clean_name, article_number)
    if is_edited:
        edited_by_source[clean_name] = True
    if is_manual:
        manual_by_source[clean_name] = True
    if edited_at:
        _update_edited_at(edited_at_by_source, clean_name, edited_at)
    if document_id is not None and clean_name not in document_ids_by_source:
        document_ids_by_source[clean_name] = document_id
    if ch is not None:
        content_hashes_by_source.setdefault(clean_name, [])
        if ch not in content_hashes_by_source[clean_name]:
            content_hashes_by_source[clean_name].append(ch)


def _filter_sources_by_min_score(
    sources: list[dict],
    min_score: float | None,
    scores_by_source: dict[str, float],
) -> list[dict]:
    if min_score is not None and sources and scores_by_source:
        filtered = [s for s in sources if s.get("max_score", 0.0) >= min_score]
        if filtered:
            return filtered
        # All sources below threshold — keep the best one so the answer
        # always has at least one source reference.
        return [sources[0]]
    return sources


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
    edited_by_source: dict[str, bool] = {}
    manual_by_source: dict[str, bool] = {}
    edited_at_by_source: dict[str, str | None] = {}
    document_ids_by_source: dict[str, int | None] = {}
    content_hashes_by_source: dict[str, list[str]] = {}

    for item in docs:
        _aggregate_source_metadata(
            item,
            pages_by_source,
            scores_by_source,
            articles_by_source,
            edited_by_source,
            manual_by_source,
            edited_at_by_source,
            document_ids_by_source,
            content_hashes_by_source,
        )

    sources = []
    for src, pages in pages_by_source.items():
        entry = _build_source_entry(
            src,
            pages,
            articles_by_source,
            scores_by_source,
            edited_by_source,
            manual_by_source,
            edited_at_by_source,
            document_ids_by_source,
            content_hashes_by_source,
        )
        sources.append(entry)

    if scores_by_source:
        sources.sort(key=lambda s: s.get("max_score", 0.0), reverse=True)

    return _filter_sources_by_min_score(sources, min_score, scores_by_source)


# ---------------------------------------------------------------------------
# Relevance gate (Self-RAG-lite)
# ---------------------------------------------------------------------------

RELEVANCE_SYSTEM = (
    "Оцени, достаточно ли предоставленного контекста для ответа на вопрос пользователя.\n"
    "Отвечай СТРОКОЙ в формате: ДА или НЕТ\n"
    "Если НЕТ — кратко укажи причину (1 предложение).\n"
    "Не отвечай на сам вопрос — только оцени достаточность контекста."
)

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RELEVANCE_SYSTEM),
        ("human", "Вопрос: {question}\n\nКонтекст из документов:\n{context}"),
    ]
)


def _get_rag_instructor_client():
    """Create instructor client for relevance checks (Ollama or OpenRouter)."""
    from infrastructure.ml.instructor_client import create_llm_instructor_client

    client, _model = create_llm_instructor_client()
    return client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True,
)
async def check_relevance(llm, question: str, docs: list) -> tuple[bool, str]:
    """Semantic check: check if the retrieved context is sufficient to answer the question.

    Uses structured output via instructor for reliable parsing.
    Returns (is_relevant, reason).  If docs is empty, returns (False, ...).
    """
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
    """Фильтрует источники, оставляя только те, на которые LLM действительно ссылалась.

    Ищет паттерны [N] в ответе и возвращает источники с соответствующими индексами (1-based).
    Если в ответе нет ни одной ссылки или ни один индекс не совпадает — возвращаем все источники.
    """
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    if not cited:
        return sources
    filtered = [src for i, src in enumerate(sources, 1) if i in cited]
    return filtered if filtered else sources
