"""Render selected context blocks with sanitized headers and stable citations."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from langchain_core.documents import Document

from domain.knowledge.enums import PageContentType
from domain.rag.evidence.context import content_with_parent_scope
from domain.rag.guardrails.sanitization import sanitize_for_prompt
from infrastructure.ml.rag.document_sources import group_by_document
from infrastructure.ml.rag.utils import clean_source_name


MAX_SECTION_LENGTH = 80

CONTEXT_SEPARATOR = "\n\n---\n\n"


def build_header(doc: Document, index: int) -> str:  # noqa: C901
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
        truncated = (
            section if len(section) <= MAX_SECTION_LENGTH else section[: MAX_SECTION_LENGTH - 3] + "..."
        )
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


def render_selected_docs(
    docs: list[Document | tuple[Document, float]],
) -> tuple[str, list[Document | tuple[Document, float]]]:
    """Group an already chosen set for display and assign stable source citations."""
    parts: list[str] = []
    selected: list[Document | tuple[Document, float]] = []

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
