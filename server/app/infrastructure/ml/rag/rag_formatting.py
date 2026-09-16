"""RAG formatting — document formatting for prompts and history conversion."""

import logging

from domain.services.rag_policy import sanitize_for_prompt
from domain.value_objects.message_role import MessageRole
from domain.value_objects.page_content_type import PageContentType
from infrastructure.ml.rag.utils import clean_source_name as _clean_source_name
from langchain_core.messages import AIMessage, HumanMessage

log = logging.getLogger("default")

# Approximate tokens per character for Russian text (~4 chars per token)
CHARS_PER_TOKEN = 4


def _build_header(doc, index: int) -> str:
    """Build a formatted header line for a document chunk."""
    source = doc.metadata.get("source", "unknown")
    source_name = _clean_source_name(source)
    parts = [f"[{index}] {source_name}"]

    doc_title = doc.metadata.get("doc_title")
    if doc_title and doc_title != source_name:
        parts.append(doc_title)

    doc_type = doc.metadata.get("doc_type")
    if doc_type:
        parts.append(f"[{doc_type}]")

    section = doc.metadata.get("section")
    if section:
        truncated = section if len(section) <= 80 else section[:77] + "..."
        parts.append(truncated)

    content_type = doc.metadata.get("content_type")
    if content_type == PageContentType.TABLE.value:
        parts.append("(таблица)")

    doc_date = doc.metadata.get("doc_date")
    if doc_date:
        parts.append(f"от {doc_date}")

    page = doc.metadata.get("page")
    page_start = doc.metadata.get("page_start")
    page_end = doc.metadata.get("page_end")
    article_number = doc.metadata.get("article_number")
    if article_number:
        parts[-1] += f", ст. {article_number}"
    elif page_start is not None and page_end is not None and page_start != page_end:
        parts[-1] += f" (стр. {page_start}-{page_end})"
    elif page is not None:
        parts[-1] += f" (стр. {page})"

    return " | ".join(parts)


def format_docs(docs, max_context_tokens: int = 6000) -> str:
    """Форматирует найденные чанки в строку для промпта.

    Принимает list[Document] или list[tuple[Document, float]] (после rerank_documents).
    Respects context budget: truncates docs list if total estimated tokens exceed limit.
    """
    parts: list[str] = []
    total_chars = 0
    max_chars = max_context_tokens * CHARS_PER_TOKEN

    for i, item in enumerate(docs, 1):
        doc = item[0] if isinstance(item, tuple) else item
        header = _build_header(doc, i)
        content = sanitize_for_prompt(doc.page_content)
        part_text = f"{header}\n{content}"
        part_chars = len(part_text)

        separator_len = 6  # "\n\n---\n\n"
        if total_chars + part_chars > max_chars and parts:
            log.warning(
                "Context budget reached: %d/%d tokens, stopping at %d docs",
                total_chars // CHARS_PER_TOKEN,
                max_context_tokens,
                len(parts),
            )
            break

        total_chars += part_chars + (separator_len if parts else 0)
        parts.append(part_text)

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
