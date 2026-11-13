"""Bounded table batches with repeated column headers and row identity."""

from collections.abc import Callable
import logging
import re
from typing import Any

from langchain.schema import Document

from domain.ingestion.content_splitter import MIN_STRUCTURAL_CHUNK_CHARS
from domain.ingestion.embed_text import context_prefix
from domain.ingestion.text_budget import fallback_split
from infrastructure.ml.ingestion.text_chunks import text_tree_chunks


log = logging.getLogger("default")


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def split_row(
    line: str,
    budget: int,
    measure: Callable[[str], int] = len,
    *,
    on_fallback: Callable[[], None] | None = None,
) -> list[str]:
    """Keep short identifying cells alongside each part of a large value."""
    cells = _cells(line)

    if not cells:
        return fallback_split([line], budget, 0, measure, on_fallback=on_fallback)

    fixed = [
        cell if measure(cell) <= (MIN_STRUCTURAL_CHUNK_CHARS if measure is len else 25) else ""
        for cell in cells
    ]

    if measure(_row([cells[0], *fixed[1:]])) < budget:
        fixed[0] = cells[0]

    if measure(_row(fixed)) >= budget or fixed == cells:
        fixed = [cells[0], *([""] * (len(cells) - 1))]

    if measure(_row(fixed)) >= budget:
        # Even the row identifier or empty column shell can exceed the limit.
        # The complete header/key are still available in chunk metadata.
        return fallback_split([line], budget, 0, measure, on_fallback=on_fallback)

    parts = []

    for index, value in enumerate(cells):
        if fixed[index] == value:
            continue

        available = budget - measure(_row(fixed))

        for piece in fallback_split([value], available, 0, measure, on_fallback=on_fallback):
            values = list(fixed)
            values[index] = piece
            parts.append(_row(values))

    return parts or [line]


def table_document_chunks(
    doc: Document,
    chunk_size: int,
    max_rows: int,
    measure: Callable[[str], int] = len,
    *,
    target: int | None = None,
    on_fallback: Callable[[], None] | None = None,
) -> list[Document]:
    """Pack rows by both count and final size, including scope and headers."""
    if chunk_size <= 0 or max_rows <= 0:
        message = "chunk_size and max_rows must be positive"
        raise ValueError(message)

    if doc.metadata.get("table_replacement_suffix"):
        return table_chunks_with_operation(
            doc, chunk_size, max_rows, measure, target=target, on_fallback=on_fallback
        )

    lines = doc.page_content.splitlines()

    if len(lines) < MIN_TABLE_LINES:
        return text_tree_chunks(doc, chunk_size, 0, measure=measure, on_fallback=on_fallback)

    # GFM delimiter cells need only one hyphen ("|-|:-:|"), with optional colons.
    header_size = 2 if re.fullmatch(r"\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?", lines[1]) else 1
    header = "\n".join(lines[:header_size])
    metadata = {**doc.metadata, "table_header": header}
    prefix = context_prefix(metadata, chunk_size, measure=measure)
    # An optional document title must not split rows that fit with their table scope.
    longest_row = max((measure(line) for line in lines[header_size:] if line.strip()), default=0)

    if measure(prefix + header + "\n") + longest_row > chunk_size:
        prefix = context_prefix({**metadata, "doc_title": ""}, chunk_size, measure=measure)

    metadata["context_prefix"] = prefix
    row_budget = chunk_size - measure(prefix + header + "\n") - (3 if measure is not len else 0)

    if row_budget <= 0:
        return text_tree_chunks(
            Document(page_content=doc.page_content, metadata=metadata),
            chunk_size,
            0,
            measure=measure,
            target=target,
            on_fallback=on_fallback,
        )

    chunks = pack_rows(
        lines[header_size:],
        metadata,
        row_budget,
        max_rows=max_rows,
        measure=measure,
        target_budget=max(1, (target or chunk_size) - measure(prefix + header + "\n") - 3),
        on_fallback=on_fallback,
    )

    # Exact final-text validation covers non-additive tokenization of Markdown joins.
    return [
        part
        for chunk in chunks
        for part in (
            [chunk]
            if measure(chunk.page_content) <= chunk_size
            else text_tree_chunks(chunk, chunk_size, 0, measure=measure, on_fallback=on_fallback)
        )
    ]


def pack_rows(
    lines: list[str],
    metadata: dict[str, Any],
    row_budget: int,
    *,
    max_rows: int,
    measure: Callable[[str], int] = len,
    target_budget: int | None = None,
    on_fallback: Callable[[], None] | None = None,
) -> list[Document]:
    """Keep ordinary rows intact; put each partial oversized row in its own batch."""
    prefix = str(metadata["context_prefix"])
    header = str(metadata["table_header"])
    chunks = []
    pending: list[str] = []
    low = high = 0

    def flush() -> None:
        if pending:
            content = prefix + header + "\n" + "\n".join(pending)
            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "table_row_start": low,
                        "table_row_end": high,
                    },
                )
            )
            pending.clear()

    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        if measure(line) > row_budget:
            flush()
            cells = _cells(line)

            chunks.extend(
                Document(
                    page_content=prefix + header + "\n" + part,
                    metadata={
                        **metadata,
                        "table_row_start": number,
                        "table_row_end": number,
                        "table_row_key": cells[0] if cells else "",
                    },
                )
                for part in split_row(line, row_budget, measure, on_fallback=on_fallback)
            )

            continue

        if pending and (
            len(pending) >= max_rows
            or measure("\n".join([*pending, line])) > min(row_budget, target_budget or row_budget)
        ):
            flush()

        if not pending:
            low = number

        high = number
        pending.append(line)

    flush()
    return chunks


MIN_TABLE_LINES = 3


def table_chunks_with_operation(
    doc: Document,
    chunk_size: int,
    max_rows: int,
    measure: Callable[[str], int],
    *,
    target: int | None,
    on_fallback: Callable[[], None] | None,
) -> list[Document]:
    """Keep a replacement operation with the last batch of the replaced rows when it fits."""
    suffix = str(doc.metadata["table_replacement_suffix"])
    metadata = {key: value for key, value in doc.metadata.items() if key != "table_replacement_suffix"}
    plain = Document(page_content=doc.page_content, metadata=metadata)
    budget = chunk_size - measure("\n" + suffix) - 2
    parts = (
        table_document_chunks(plain, budget, max_rows, measure, target=target, on_fallback=on_fallback)
        if budget > 0
        else []
    )

    if parts and measure(parts[-1].page_content + "\n" + suffix) <= chunk_size:
        parts[-1].page_content += "\n" + suffix
        return parts

    # The operation stays a separate chunk and part of the parent scope;
    # only its copy next to the replaced row is dropped.
    log.warning("Table replacement operation does not fit the chunk budget; kept separately")
    return table_document_chunks(plain, chunk_size, max_rows, measure, target=target, on_fallback=on_fallback)
