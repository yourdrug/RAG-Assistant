"""Render selected context blocks with sanitized headers and stable citations."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, cast


if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.documents import Document

from domain.knowledge.enums import PageContentType
from domain.rag.evidence.context import content_with_parent_scope
from domain.rag.guardrails.sanitization import sanitize_for_prompt
from domain.shared.breadcrumb import shorten_breadcrumb
from infrastructure.ml.ingestion.text_chunks import context_prefix
from infrastructure.ml.rag.document_sources import group_by_document
from infrastructure.ml.rag.utils import clean_source_name


MAX_SECTION_LENGTH = 80
# Shorter suffix/prefix coincidences are likely accidental, not splitter overlap.
MIN_CHUNK_OVERLAP = 30

CONTEXT_SEPARATOR = "\n\n---\n\n"


def build_header(doc: Document, index: int) -> str:
    """Build a formatted header line for a document chunk.

    All metadata fields are sanitized via ``sanitize_for_prompt`` to prevent
    injection through attacker-controlled filenames, titles, or section names.
    """
    source = doc.metadata.get("source", "unknown")
    source_name = sanitize_for_prompt(clean_source_name(source))
    doc_title = doc.metadata.get("doc_title")
    parts = [f"[{index}] {sanitize_for_prompt(doc_title) if doc_title else source_name}"]

    doc_type = doc.metadata.get("doc_type")

    if doc_type:
        parts.append(f"[{sanitize_for_prompt(doc_type)}]")

    section = doc.metadata.get("section")

    if section:
        truncated = shorten_breadcrumb(section, MAX_SECTION_LENGTH)
        parts.append(sanitize_for_prompt(truncated))

    content_type = doc.metadata.get("content_type")

    if content_type == PageContentType.TABLE.value:
        parts.append("(таблица)")

    doc_date = doc.metadata.get("doc_date")

    if doc_date:
        parts.append(f"от {sanitize_for_prompt(str(doc_date))}")

    append_location(parts, doc)

    return " | ".join(parts)


def render_selected_docs[ItemT: Document | tuple[Document, float]](
    docs: Sequence[ItemT],
) -> tuple[str, list[ItemT]]:
    """Group an already chosen set for display and assign stable source citations."""
    parts: list[str] = []
    selected: list[ItemT] = []

    for citation_id, group in enumerate(group_by_document(docs), start=1):

        def position_key(item: ItemT) -> tuple[bool, int, int]:
            doc = item[0] if isinstance(item, tuple) else item
            index = doc.metadata.get("chunk_index")
            return (
                not isinstance(index, int),
                int(doc.metadata.get("act_version_id") or 0) if isinstance(index, int) else 0,
                index if isinstance(index, int) else 0,
            )

        ordered_group = sorted(group, key=position_key)
        scopes: set[str] = set()
        previous = None
        previous_body = ""

        for position, item in enumerate(ordered_group):
            doc = item[0] if isinstance(item, tuple) else item
            header = build_header(doc, citation_id)

            if position:
                header = header.split("] ", 1)[1]

            display = copy(doc)
            body = chunk_body(doc)
            display.page_content = body

            if (
                previous is not None
                and isinstance(doc.metadata.get("chunk_index"), int)
                and previous.metadata.get("chunk_index") == doc.metadata["chunk_index"] - 1
                and doc.metadata.get("document_id") is not None
                and doc.metadata.get("document_id") == previous.metadata.get("document_id")
                and doc.metadata.get("act_version_id") == previous.metadata.get("act_version_id")
            ):
                display.page_content = body[chunk_overlap_length(previous_body, body) :]

            heading = removed_unit_heading(doc)

            if heading and not display.page_content.lstrip().startswith(heading.rstrip("…")):
                display.page_content = f"{heading}\n{display.page_content}"

            previous, previous_body = doc, body
            content = sanitize_for_prompt(content_with_parent_scope(display, scopes))
            parts.append(f"{header}\n{content}")
            selected_doc = copy(doc)
            selected_doc.metadata = {**doc.metadata, "citation_id": citation_id}
            selected.append(
                cast("ItemT", (selected_doc, item[1]) if isinstance(item, tuple) else selected_doc)
            )

    return CONTEXT_SEPARATOR.join(parts), selected


def chunk_overlap_length(previous: str, current: str) -> int:
    """Length of the longest suffix of ``previous`` that is a prefix of ``current``.

    Linear-time prefix function (KMP); runs for every trial render on the chat path.
    """
    tail = previous[-len(current) :] if current else ""
    text = f"{current}\x00{tail}"
    border = [0] * len(text)

    for index in range(1, len(text)):
        length = border[index - 1]

        while length and text[index] != text[length]:
            length = border[length - 1]

        if text[index] == text[length]:
            length += 1

        border[index] = length

    overlap = min(border[-1] if text else 0, len(current), len(tail))
    # A NUL inside chunk text could let the border cross the separator; recheck.
    return overlap if overlap >= MIN_CHUNK_OVERLAP and previous.endswith(current[:overlap]) else 0


def append_location(parts: list[str], doc: Document) -> None:
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


def ingestion_prefix(doc: Document) -> str:
    """Return the exact ingestion prefix present at the start of the chunk, if any."""
    body = doc.page_content
    prefix = doc.metadata.get("context_prefix", "")

    if prefix:
        return prefix if body.startswith(prefix) else ""

    for include_unit in (True, False):
        candidate = context_prefix(doc.metadata, max(len(body), 360), include_unit=include_unit)

        if candidate and body.startswith(candidate):
            return candidate

    return ""


def chunk_body(doc: Document) -> str:
    """Remove only an exact ingestion prefix, retaining original chunk metadata."""
    return doc.page_content.removeprefix(ingestion_prefix(doc))


def removed_unit_heading(doc: Document) -> str:
    """Return the provision label that ``chunk_body`` stripped with the prefix.

    ``content_with_parent_scope`` restores parent units but not the unit heading
    ("Статья 12. ..."), which the model needs to identify and cite the provision.
    """
    heading = doc.metadata.get("unit_heading_text")
    prefix = ingestion_prefix(doc)

    if not heading or not prefix:
        return ""

    # Current prefixes put the label inside a path; legacy prefixes used a line.
    # A section label is already restored by build_header.
    if heading in (doc.metadata.get("section") or ""):
        return ""

    for line in prefix.splitlines():
        for label in line.split(" > "):
            if label == heading or (label.endswith("…") and heading.startswith(label[:-1])):
                return label

    return ""
