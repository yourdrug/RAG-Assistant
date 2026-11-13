"""Bounded text chunks with structural context and original page provenance."""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from domain.ingestion.content_splitter import TINY_FRAGMENT_MAX_CHARS
from domain.ingestion.embed_text import context_prefix
from domain.ingestion.profiles.patterns import SENTENCE_RE


if TYPE_CHECKING:
    from collections.abc import Callable


STEP_HEADING_RE = re.compile(r"^[ \t]*(?:Шаг\s+)?(\d+)[.)][ \t]+([^\n]+)", re.MULTILINE)


def split_at_boundaries(text: str, patterns: list[re.Pattern[str]]) -> list[str]:
    """Slice at match offsets, retaining complete markers and capture groups."""
    offsets = sorted({0, len(text), *(m.start() for pattern in patterns for m in pattern.finditer(text))})
    return [text[start:end] for start, end in itertools.pairwise(offsets) if text[start:end].strip()]


def sentence_fragments(text: str) -> list[str]:
    """Keep exact source spans and the legal abbreviation dictionary."""
    offsets = [0, *(match.end() for match in SENTENCE_RE.finditer(text)), len(text)]
    return [text[start:end] for start, end in itertools.pairwise(offsets) if text[start:end].strip()]


def overlap_tail(text: str, size: int, measure: Callable[[str], int] = len) -> str:
    """Return complete trailing sentences whose total size fits the overlap."""
    tail = ""

    for fragment in reversed(sentence_fragments(text)):
        candidate = fragment + tail

        if measure(candidate) > size:
            break

        tail = candidate

    return tail if size > 0 else ""


def merge_fragments(
    fragments: list[str], chunk_size: int, chunk_overlap: int, measure: Callable[[str], int] = len
) -> list[str]:
    """Pack exact contiguous fragments, overlapping only complete sentences."""
    merged: list[str] = []
    current = ""

    for fragment in fragments:
        if current and measure(current + fragment) > chunk_size:
            merged.append(current.strip())
            current = overlap_tail(current, chunk_overlap, measure)

            if measure(current + fragment) > chunk_size:
                current = ""

        current += fragment

    if current.strip():
        merged.append(current.strip())

    return merged


def fallback_split(
    chunks: list[str],
    chunk_size: int,
    chunk_overlap: int,
    measure: Callable[[str], int] = len,
    *,
    on_fallback: Callable[[], None] | None = None,
) -> list[str]:
    """Split whole sentences first; only oversized sentences reach word/character splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=measure,
        separators=[r"\s+", ""],
        is_separator_regex=True,
        keep_separator="end",
    )
    result = []

    for chunk in chunks:
        pending: list[str] = []

        for sentence in sentence_fragments(chunk):
            if measure(sentence) <= chunk_size:
                pending.append(sentence)
                continue

            result.extend(merge_fragments(pending, chunk_size, chunk_overlap, measure))
            pending = []

            if on_fallback:
                on_fallback()

            result.extend(splitter.split_text(sentence))

        result.extend(merge_fragments(pending, chunk_size, chunk_overlap, measure))

    return result


def split_overflow(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    boundary_patterns: list[re.Pattern[str]] | None = None,
    measure: Callable[[str], int] = len,
    *,
    on_fallback: Callable[[], None] | None = None,
) -> list[str]:
    """Retain structural markers while enforcing a strict character budget."""
    if measure(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    if boundary_patterns:
        fragments = split_at_boundaries(text, boundary_patterns)
        chunks = merge_fragments(fragments, chunk_size, chunk_overlap, measure)
    else:
        chunks = [text]

    parts = fallback_split(chunks, chunk_size, chunk_overlap, measure, on_fallback=on_fallback)
    result = []
    index = 0

    while index < len(parts):
        part = parts[index]

        if (
            len(part) < TINY_FRAGMENT_MAX_CHARS
            and index + 1 < len(parts)
            and re.fullmatch(r"(?:\d+[.\d]*|Статья\s+\d+[.\d]*)", part)
        ):
            if parts[index + 1].startswith(part):
                # The next body already carries this marker through overlap.
                index += 1
                continue

            prefix = part + "\n"
            budget = chunk_size - measure(prefix)

            if budget > 0:
                following = fallback_split(
                    [parts[index + 1]],
                    budget,
                    min(chunk_overlap, budget - 1),
                    measure,
                    on_fallback=on_fallback,
                )
                result.extend([prefix + following[0], *following[1:]])
                index += 2
            else:
                result.append(part)
                index += 1
        else:
            result.append(part)
            index += 1

    return result


def step_documents(doc: Document) -> list[Document]:
    """Treat numbered steps and their bodies as atomic blocks when they fit."""
    matches = list(STEP_HEADING_RE.finditer(doc.page_content))

    if not matches:
        return [doc]

    result = []
    intro = doc.page_content[: matches[0].start()].strip()
    parents = list(doc.metadata.get("parent_units", []))

    if intro:
        result.append(document_slice(doc, intro))
        parents.append(
            {
                "heading": doc.metadata.get("section") or "Instruction introduction",
                "unit_kind": "preamble",
                "boundary_value": None,
                "content": intro,
            }
        )

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc.page_content)
        heading = match.group().strip()
        parent = doc.metadata.get("section")
        step = document_slice(doc, doc.page_content[match.start() : end].strip(), match.start())
        step.metadata.update(
            section=f"{parent} > {heading}" if parent else heading,
            heading=heading,
            unit_heading_text=heading,
            step_number=match.group(1),
        )

        if parents:
            step.metadata["parent_units"] = parents

        result.append(step)

    return result


def pack_step_documents(
    doc: Document, chunk_size: int, measure: Callable[[str], int] = len
) -> list[Document]:
    """Pack complete neighboring steps, retaining all their numbers."""
    steps = step_documents(doc)

    if len(steps) == 1:
        return steps

    budget = chunk_size - measure(context_prefix(doc.metadata, chunk_size, measure=measure))
    result: list[Document] = []
    pending: list[Document] = []
    low = high = 0

    for step in steps:
        located = locate_span(doc.page_content, step.page_content, high)

        if located is None:
            message = "Cannot locate instruction step in its source"
            raise ValueError(message)

        start, end = located

        if pending and measure(doc.page_content[low:end]) > budget:
            result.append(_packed_steps(doc, pending, low, high))
            pending = []

        if not pending:
            low = start

        pending.append(step)
        high = end

    if pending:
        result.append(_packed_steps(doc, pending, low, high))

    return result


def _packed_steps(doc: Document, steps: list[Document], start: int, end: int) -> Document:
    if len(steps) == 1:
        return steps[0]

    merged = document_slice(doc, doc.page_content[start:end], start)
    merged.metadata["step_numbers"] = [
        step.metadata["step_number"] for step in steps if "step_number" in step.metadata
    ]
    parents = list(merged.metadata.get("parent_units", []))

    for step in steps:
        for parent in step.metadata.get("parent_units", []):
            if parent not in parents:
                parents.append(parent)

    if parents:
        merged.metadata["parent_units"] = parents

    return merged


def locate_span(source: str, content: str, cursor: int = 0) -> tuple[int, int] | None:
    """Locate text even if structural merging changed only whitespace."""
    start = source.find(content, cursor)

    if start >= 0:
        return start, start + len(content)

    pattern = r"\s+".join(re.escape(word) for word in content.split())

    if not pattern:
        return None

    match = re.search(pattern, source[cursor:])
    return (cursor + match.start(), cursor + match.end()) if match else None


def document_slice(doc: Document, content: str, cursor: int = 0) -> Document:
    """Carry page offsets relative to a structural slice of the source."""
    meta = dict(doc.metadata)
    spans = meta.get("_page_spans")

    if spans:
        located = locate_span(doc.page_content, content, cursor)

        if located is None:
            message = "Cannot map structural unit to its source pages"
            raise ValueError(message)

        start, end = located
        meta["_page_spans"] = [
            (max(low, start) - start, min(high, end) - start, page)
            for low, high, page in spans
            if low < end and high > start
        ]

    return Document(page_content=content, metadata=meta)


def assign_chunk_pages(
    chunks: list[Document],
    source: Document,
    bodies: list[str],
    overlap: int,
    measure: Callable[[str], int] = len,
) -> None:
    """Derive each final chunk's pages from its body, excluding repeated context."""
    spans = source.metadata.get("_page_spans")

    if not spans:
        return

    cursor = 0

    for chunk, body in zip(chunks, bodies, strict=True):
        located = locate_span(source.page_content, body, cursor)

        if located is None:
            message = "Cannot map chunk body to its source pages"
            raise ValueError(message)

        start, end = located
        cursor = max(start + 1, end - len(overlap_tail(body, overlap, measure)))
        pages = sorted({page for low, high, page in spans if low < end and high > start})
        chunk.metadata.pop("_page_spans", None)
        chunk.metadata.pop("page", None)
        chunk.metadata.update(page_start=pages[0], page_end=pages[-1], pages=pages)

        if len(pages) == 1:
            chunk.metadata["page"] = pages[0]


def text_document_chunks(
    doc: Document,
    chunk_size: int,
    chunk_overlap: int,
    patterns: list[re.Pattern[str]] | None = None,
    measure: Callable[[str], int] = len,
    *,
    target: int | None = None,
    on_fallback: Callable[[], None] | None = None,
) -> list[Document]:
    """Fit atomic bodies to max; split oversized bodies near target in measured units."""
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        message = "chunk_size must be positive and 0 <= chunk_overlap < chunk_size"
        raise ValueError(message)

    prefix = context_prefix(doc.metadata, chunk_size, body=doc.page_content, measure=measure)
    include_unit = measure(prefix + doc.page_content) > chunk_size

    if include_unit:
        prefix = context_prefix(doc.metadata, chunk_size, include_unit=True, measure=measure)

    body_budget = max(1, chunk_size - measure(prefix) - (2 if measure is not len else 0))
    packing_budget = min(body_budget, max(1, (target or chunk_size) - measure(prefix)))
    overlap = min(chunk_overlap, packing_budget - 1)

    if measure(prefix + doc.page_content) <= chunk_size:
        bodies = [doc.page_content.strip()]
    else:
        bodies = split_overflow(
            doc.page_content, packing_budget, overlap, patterns, measure, on_fallback=on_fallback
        )

    bodies = [body for body in bodies if any(char.isalnum() for char in body)]
    heading = doc.metadata.get("unit_heading_text") or doc.metadata.get("heading")

    if heading and heading in prefix and len(bodies) > 1:
        bodies = [body for body in bodies if body.strip() != heading]

    chunks = []
    final_bodies = []

    for body in bodies:
        body_prefix = context_prefix(
            doc.metadata, chunk_size, include_unit=include_unit, body=body, measure=measure
        )
        # Check actual concatenation: token counts are not exactly additive at joins.
        available = max(1, chunk_size - measure(body_prefix) - 2)
        pieces = (
            [body]
            if measure(body_prefix + body) <= chunk_size
            else fallback_split([body], available, 0, measure, on_fallback=on_fallback)
        )

        for piece in pieces:
            if measure(body_prefix + piece) > chunk_size:
                message = "Context and source body exceed the chunk token budget"
                raise ValueError(message)

            chunks.append(
                Document(
                    page_content=body_prefix + piece,
                    metadata={**doc.metadata, "context_prefix": body_prefix},
                )
            )
            final_bodies.append(piece)

    assign_chunk_pages(chunks, doc, final_bodies, overlap, measure)
    return chunks
