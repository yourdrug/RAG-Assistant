"""Bounded text chunks with structural context and original page provenance."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import TYPE_CHECKING

from langchain.schema import Document

from domain.ingestion.chunking_policy import ChunkingPolicy
from domain.ingestion.document_tree import DocNode, NodeKind
from domain.ingestion.embed_text import context_prefix
from domain.ingestion.text_budget import fallback_split, locate_span, overlap_tail
from domain.ingestion.tree_packer import pack_tree


if TYPE_CHECKING:
    from collections.abc import Callable


STEP_HEADING_RE = re.compile(r"^[ \t]*(?:Шаг\s+)?(\d+)[.)][ \t]+([^\n]+)", re.MULTILINE)


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


def pack_instruction_tree(
    doc: Document,
    chunk_size: int,
    measure: Callable[[str], int] = len,
    *,
    policy: ChunkingPolicy | None = None,
) -> list[Document]:
    """Instruction steps are atomic tree siblings, packed by the common policy."""
    steps = step_documents(doc)

    if len(steps) == 1:
        return steps

    budget = max(1, chunk_size - measure(context_prefix(doc.metadata, chunk_size, measure=measure)))
    nodes = []
    cursor = 0

    for step in steps:
        located = locate_span(doc.page_content, step.page_content, cursor)

        if located is None:
            message = "Cannot locate instruction step in its source"
            raise ValueError(message)

        start, end = located
        nodes.append(
            DocNode(
                NodeKind.STEP if "step_number" in step.metadata else NodeKind.PREAMBLE,
                number=step.metadata.get("step_number"),
                text=step.page_content,
                span=(start, end),
            )
        )
        cursor = end

    tree = DocNode(
        NodeKind.DOCUMENT,
        children=tuple(nodes),
        span=(0, len(doc.page_content)),
        hints={"source_text": doc.page_content},
    )
    drafts = pack_tree(
        tree,
        replace(policy, target_tokens=budget, max_tokens=budget, overlap_tokens=0)
        if policy
        else ChunkingPolicy(budget, budget),
        measure,
        split_leaves=False,
    )
    return [
        _packed_steps(
            doc,
            [
                step
                for step, node in zip(steps, nodes, strict=True)
                if draft.span[0] <= node.span[0] and node.span[1] <= draft.span[1]
            ],
            *draft.span,
        )
        for draft in drafts
    ]


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


def document_slice(doc: Document, content: str, cursor: int = 0) -> Document:
    """Carry page offsets relative to a structural slice of the source."""
    meta = dict(doc.metadata)
    spans = meta.get("_page_spans")

    if spans or meta.get("line_hints"):
        located = locate_span(doc.page_content, content, cursor)

        if located is None:
            message = "Cannot map structural unit to its source pages"
            raise ValueError(message)

        start, end = located
        meta["_page_spans"] = [
            (max(low, start) - start, min(high, end) - start, page)
            for low, high, page in (spans or [])
            if low < end and high > start
        ]

        if not spans:
            meta.pop("_page_spans", None)

        meta["line_hints"] = [
            {**hint, "start": hint["start"] - start, "end": min(hint["end"], end) - start}
            for hint in meta.get("line_hints", [])
            if start <= hint["start"] < end
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


def text_tree_chunks(  # noqa: PLR0915 - context reservation and final provenance are one operation
    doc: Document,
    chunk_size: int,
    chunk_overlap: int,
    _patterns: list[re.Pattern[str]] | None = None,
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
        tree = DocNode(
            NodeKind.PARAGRAPH,
            text=doc.page_content,
            span=(0, len(doc.page_content)),
            hints={"source_text": doc.page_content},
        )
        bodies = [
            draft.content
            for draft in pack_tree(
                tree,
                ChunkingPolicy(packing_budget, packing_budget, overlap_tokens=overlap),
                measure,
                on_fallback=on_fallback,
            )
        ]

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
