"""RAG formatting — document formatting for prompts and history conversion."""

from domain.chat.message import MessageRole
from domain.rag.guardrails.sanitization import sanitize_for_prompt
from langchain_core.messages import AIMessage, HumanMessage

from infrastructure.ml.rag.context_rendering import (  # noqa: F401
    build_header,
    render_selected_docs,
)
from infrastructure.ml.rag.context_selection import select_context
from infrastructure.ml.rag.prompt_budget import estimate_text_tokens

# Legacy neighbor-prefetch estimate; prompt selection uses prompt_budget counters.
CHARS_PER_TOKEN = 4


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
