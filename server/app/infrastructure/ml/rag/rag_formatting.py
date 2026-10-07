"""RAG formatting — document formatting for prompts and history conversion."""

from copy import copy

from domain.services.rag_policy import sanitize_for_prompt
from domain.value_objects.message_role import MessageRole
from domain.value_objects.page_content_type import PageContentType
from infrastructure.ml.rag.utils import clean_source_name
from infrastructure.ml.rag.document_sources import group_by_document
from infrastructure.ml.rag.prompt_budget import estimate_text_tokens
from langchain_core.messages import AIMessage, HumanMessage

# Legacy neighbor-prefetch estimate; prompt selection uses prompt_budget counters.
CHARS_PER_TOKEN = 4
CONTEXT_SEPARATOR = "\n\n---\n\n"


def build_header(doc, index: int) -> str:
    """Build a formatted header line for a document chunk.

    All metadata fields are sanitized via ``sanitize_for_prompt`` to prevent
    injection through attacker-controlled filenames, titles, or section names.
    """
    source = doc.metadata.get("source", "unknown")
    source_name = sanitize_for_prompt(clean_source_name(source))
    parts = [f"[{index}] {source_name}"]

    doc_title = doc.metadata.get("doc_title")
    if doc_title and doc_title != source_name:
        parts.append(sanitize_for_prompt(doc_title))

    doc_type = doc.metadata.get("doc_type")
    if doc_type:
        parts.append(f"[{sanitize_for_prompt(doc_type)}]")

    section = doc.metadata.get("section")
    if section:
        truncated = section if len(section) <= 80 else section[:77] + "..."
        parts.append(sanitize_for_prompt(truncated))

    content_type = doc.metadata.get("content_type")
    if content_type == PageContentType.TABLE.value:
        parts.append("(таблица)")

    doc_date = doc.metadata.get("doc_date")
    if doc_date:
        parts.append(f"от {sanitize_for_prompt(str(doc_date))}")

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


def content_with_parent_scope(doc) -> str:
    """Include full structural conditions that did not fit inside the chunk."""
    parts = []
    normalized = " ".join(doc.page_content.split())
    scopes = [parent.get("content", "") for parent in doc.metadata.get("parent_units", [])]
    if doc.metadata.get("content_type") == PageContentType.TABLE.value:
        scopes.extend(doc.metadata.get(key, "") for key in ("table_header", "table_row_key"))
    for scope_text in scopes:
        content = scope_text.strip()
        scope = " ".join(content.split())
        if scope and scope not in normalized:
            parts.append(content)
            normalized += " " + scope
    parts.append(doc.page_content)
    return "\n".join(parts)


def render_selected_docs(docs) -> tuple[str, list]:
    """Group an already chosen set for display and assign stable source citations."""
    parts: list[str] = []
    selected: list = []
    for citation_id, group in enumerate(group_by_document(docs), start=1):
        for position, item in enumerate(group):
            doc = item[0] if isinstance(item, tuple) else item
            header = build_header(doc, citation_id)
            if position:
                header = header.split("] ", 1)[1]
            content = sanitize_for_prompt(content_with_parent_scope(doc))
            parts.append(f"{header}\n{content}")
            selected_doc = copy(doc)
            selected_doc.metadata = {**doc.metadata, "citation_id": citation_id}
            selected.append((selected_doc, item[1]) if isinstance(item, tuple) else selected_doc)
    return CONTEXT_SEPARATOR.join(parts), selected


def format_docs_with_selection(
    docs,
    max_context_tokens: int = 6000,
    *,
    context_counter=estimate_text_tokens,
    query_parts: list[str] | None = None,
    exclusions: list[dict] | None = None,
) -> tuple[str, list]:
    """Select complete scoped chunks before grouping them for presentation.

    ``context_counter`` may count a complete prompt containing the trial context.
    The caller then supplies the full input budget, including message framing.
    """
    from infrastructure.ml.rag.context_selection import select_context

    return select_context(docs, max_context_tokens, context_counter, query_parts or [], exclusions)


def format_docs(docs, max_context_tokens: int = 6000) -> str:
    """Format retrieved documents into prompt context within the token budget."""
    formatted, _ = format_docs_with_selection(docs, max_context_tokens)
    return formatted


def history_to_messages(history: list[dict]):
    """Convert history from DB to LangChain messages.

    All message content is sanitized via ``sanitize_for_prompt`` to prevent
    multi-turn injection through stored conversation history.
    """
    messages: list[HumanMessage | AIMessage] = []
    for msg in history:
        content = sanitize_for_prompt(msg["content"])
        if msg["role"] == MessageRole.USER.value:
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages
