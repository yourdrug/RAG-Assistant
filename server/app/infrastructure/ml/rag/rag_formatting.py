"""RAG formatting — document formatting for prompts and history conversion."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from domain.chat.message import MessageRole
from domain.rag.guardrails.sanitization import sanitize_for_prompt
from infrastructure.ml.rag.context_rendering import (
    build_header,
    render_selected_docs,
)
from infrastructure.ml.rag.context_selection import select_context
from infrastructure.ml.rag.prompt_budget import estimate_text_tokens
from infrastructure.ml.tokenization import CHARS_PER_TOKEN


def format_docs_with_selection[ItemT: Document | tuple[Document, float]](
    docs: Sequence[ItemT],
    max_context_tokens: int = 6000,
    *,
    context_counter: Callable[[str], int] = estimate_text_tokens,
    query_parts: list[str] | None = None,
    exclusions: list[dict[str, Any]] | None = None,
) -> tuple[str, list[ItemT]]:
    """Select complete scoped chunks before grouping them for presentation.

    ``context_counter`` may count a complete prompt containing the trial context.
    The caller then supplies the full input budget, including message framing.
    """
    return select_context(docs, max_context_tokens, context_counter, query_parts or [], exclusions)


def format_docs(docs: Sequence[Document | tuple[Document, float]], max_context_tokens: int = 6000) -> str:
    """Format retrieved documents into prompt context within the token budget."""
    formatted, _ = format_docs_with_selection(docs, max_context_tokens)
    return formatted


def history_to_messages(history: list[dict[str, Any]]) -> list[HumanMessage | AIMessage]:
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


__all__ = [
    "CHARS_PER_TOKEN",
    "build_header",
    "format_docs",
    "format_docs_with_selection",
    "history_to_messages",
    "render_selected_docs",
]
