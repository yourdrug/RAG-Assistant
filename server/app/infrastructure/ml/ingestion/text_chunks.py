"""Bounded text chunks with structural context and original page provenance."""

from __future__ import annotations

import re

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from domain.domain_profile.content_splitter import MIN_STRUCTURAL_CHUNK_CHARS, TINY_FRAGMENT_MAX_CHARS

MAX_CONTEXT_CHARS = 300
STEP_HEADING_RE = re.compile(r"^[ \t]*(?:Шаг\s+)?(\d+)[.)][ \t]+([^\n]+)", re.MULTILINE)
TEXT_SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ])", r"\s+", ""]


def split_at_boundaries(text: str, patterns: list) -> list[str]:
    """Slice at match offsets, retaining complete markers and capture groups."""
    offsets = sorted({0, len(text), *(m.start() for pattern in patterns for m in pattern.finditer(text))})
    return [
        text[start:end] for start, end in zip(offsets, offsets[1:], strict=False) if text[start:end].strip()
    ]


def overlap_tail(text: str, size: int) -> str:
    """Keep complete trailing words, rather than a suffix starting mid-word."""
    if size <= 0:
        return ""
    start = max(0, len(text) - size)
    if start and not text[start - 1].isspace():
        match = re.search(r"\s+", text[start:])
        if match is None:
            return ""
        start += match.end()
    return text[start:]


def merge_fragments(fragments: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Pack contiguous fragments without altering source wording or whitespace."""
    merged: list[str] = []
    current = ""
    for fragment in fragments:
        if current and len(current) + len(fragment) > chunk_size:
            # A fragment shorter than the overlap would be emitted twice at
            # the same source offset. Keep it attached to the following body.
            if len(current) > chunk_overlap:
                merged.append(current.strip())
                current = overlap_tail(current, chunk_overlap)
        current += fragment
    if current.strip():
        merged.append(current.strip())
    return merged


def fallback_split(chunks: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Enforce the body budget, preferring paragraph and sentence boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=TEXT_SEPARATORS,
        is_separator_regex=True,
        keep_separator="end",
    )
    return [part for chunk in chunks for part in splitter.split_text(chunk)]


def split_overflow(text: str, chunk_size: int, chunk_overlap: int, boundary_patterns: list | None = None):
    """Retain structural markers while enforcing a strict character budget."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    if boundary_patterns:
        fragments = split_at_boundaries(text, boundary_patterns)
        chunks = merge_fragments(fragments, chunk_size, chunk_overlap)
    else:
        chunks = [text]
    parts = fallback_split(chunks, chunk_size, chunk_overlap)
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
            budget = chunk_size - len(prefix)
            if budget > 0:
                following = fallback_split([parts[index + 1]], budget, min(chunk_overlap, budget - 1))
                result.extend([prefix + following[0], *following[1:]])
                index += 2
            else:
                result.append(part)
                index += 1
        else:
            result.append(part)
            index += 1
    return result


def shorten_context(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def context_prefix(metadata: dict, chunk_size: int, *, include_unit: bool = False) -> str:
    """Repeat complete parent conditions when they fit, never partial clauses.

    Full parents remain in metadata even when they cannot fit alongside a
    useful body. Prompt formatting supplies that complete scope in that case.
    """
    budget = min(MAX_CONTEXT_CHARS, chunk_size // 3)
    parts = []
    section = metadata.get("section")
    if section and budget > len("[Раздел: ]\n"):
        label_budget = min(budget, 120)
        parts.append(f"[Раздел: {shorten_context(section, label_budget - len('[Раздел: ]'))}]")
    body_reserve = min(MIN_STRUCTURAL_CHUNK_CHARS, chunk_size // 2)
    remaining = chunk_size - body_reserve - sum(len(part) + 1 for part in parts)
    for parent in reversed(metadata.get("parent_units", [])):
        content = parent.get("content", "").strip()
        if content and len(content) + 1 <= remaining:
            parts.append(content)
            remaining -= len(content) + 1
    if include_unit and metadata.get("unit_heading_text"):
        used = sum(len(part) + 1 for part in parts)
        heading_budget = min(budget - used, remaining)
        heading = metadata["unit_heading_text"]
        if heading_budget > 10 and heading not in "\n".join(parts):
            parts.append(shorten_context(heading, heading_budget - 1))
    return "\n".join(parts) + "\n" if parts else ""


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


def pack_step_documents(doc: Document, chunk_size: int) -> list[Document]:
    """Pack complete neighboring steps, retaining all their numbers."""
    steps = step_documents(doc)
    if len(steps) == 1:
        return steps
    budget = chunk_size - len(context_prefix(doc.metadata, chunk_size))
    result: list[Document] = []
    pending: list[Document] = []
    low = high = 0
    for step in steps:
        located = locate_span(doc.page_content, step.page_content, high)
        if located is None:
            raise ValueError("Cannot locate instruction step in its source")
        start, end = located
        if pending and end - low > budget:
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
            raise ValueError("Cannot map structural unit to its source pages")
        start, end = located
        meta["_page_spans"] = [
            (max(low, start) - start, min(high, end) - start, page)
            for low, high, page in spans
            if low < end and high > start
        ]
    return Document(page_content=content, metadata=meta)


def assign_chunk_pages(chunks: list[Document], source: Document, bodies: list[str], overlap: int) -> None:
    """Derive each final chunk's pages from its body, excluding repeated context."""
    spans = source.metadata.get("_page_spans")
    if not spans:
        return
    cursor = 0
    for chunk, body in zip(chunks, bodies, strict=True):
        located = locate_span(source.page_content, body, cursor)
        if located is None:
            raise ValueError("Cannot map chunk body to its source pages")
        start, end = located
        cursor = max(start + 1, end - overlap)
        pages = sorted({page for low, high, page in spans if low < end and high > start})
        chunk.metadata.pop("_page_spans", None)
        chunk.metadata.pop("page", None)
        chunk.metadata.update(page_start=pages[0], page_end=pages[-1], pages=pages)
        if len(pages) == 1:
            chunk.metadata["page"] = pages[0]


def text_document_chunks(doc: Document, chunk_size: int, chunk_overlap: int, patterns: list | None = None):
    """Split after reserving context; the final content always fits the limit."""
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_size must be positive and 0 <= chunk_overlap < chunk_size")
    normalized_body = " ".join(doc.page_content.split())
    context_metadata = {
        **doc.metadata,
        "parent_units": [
            parent
            for parent in doc.metadata.get("parent_units", [])
            if " ".join(parent.get("content", "").split()) not in normalized_body
        ],
    }
    prefix = context_prefix(context_metadata, chunk_size)
    if len(doc.page_content) > chunk_size - len(prefix):
        prefix = context_prefix(context_metadata, chunk_size, include_unit=True)
    body_budget = chunk_size - len(prefix)
    overlap = min(chunk_overlap, body_budget - 1)
    bodies = split_overflow(doc.page_content, body_budget, overlap, patterns)
    # A heading already repeated in the prefix needs no standalone chunk.
    heading = doc.metadata.get("unit_heading_text") or doc.metadata.get("heading")
    if heading and heading in prefix and len(bodies) > 1:
        bodies = [body for body in bodies if body.strip() != heading]
    chunks = [Document(page_content=prefix + body, metadata=dict(doc.metadata)) for body in bodies]
    assign_chunk_pages(chunks, doc, bodies, overlap)
    return chunks
