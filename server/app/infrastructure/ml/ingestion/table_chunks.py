"""Bounded table batches with repeated column headers and row identity."""

import re

from langchain.schema import Document

from domain.domain_profile.content_splitter import MIN_STRUCTURAL_CHUNK_CHARS
from infrastructure.ml.ingestion.text_chunks import context_prefix, fallback_split, text_document_chunks


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _split_row(line: str, budget: int) -> list[str]:
    """Keep short identifying cells alongside each part of a large value."""
    cells = _cells(line)
    if not cells:
        return fallback_split([line], budget, 0)
    fixed = [cell if len(cell) <= MIN_STRUCTURAL_CHUNK_CHARS else "" for cell in cells]
    if len(_row([cells[0], *fixed[1:]])) < budget:
        fixed[0] = cells[0]
    if len(_row(fixed)) >= budget or fixed == cells:
        fixed = [cells[0], *([""] * (len(cells) - 1))]
    if len(_row(fixed)) >= budget:
        # Even the row identifier or empty column shell can exceed the limit.
        # The complete header/key are still available in chunk metadata.
        return fallback_split([line], budget, 0)
    parts = []
    for index, value in enumerate(cells):
        if fixed[index] == value:
            continue
        available = budget - len(_row(fixed))
        for piece in fallback_split([value], available, 0):
            values = list(fixed)
            values[index] = piece
            parts.append(_row(values))
    return parts or [line]


def table_document_chunks(doc: Document, chunk_size: int, max_rows: int) -> list[Document]:
    """Pack rows by both count and final size, including scope and headers."""
    if chunk_size <= 0 or max_rows <= 0:
        raise ValueError("chunk_size and max_rows must be positive")
    lines = doc.page_content.splitlines()
    if len(lines) < 3:
        return text_document_chunks(doc, chunk_size, 0)
    header = "\n".join(lines[:2])
    metadata = {**doc.metadata, "table_header": header}
    prefix = context_prefix(metadata, chunk_size)
    row_budget = chunk_size - len(prefix) - len(header) - 1
    if row_budget <= 0:
        return text_document_chunks(Document(page_content=doc.page_content, metadata=metadata), chunk_size, 0)
    return _pack_rows(lines[2:], prefix, header, metadata, row_budget, max_rows)


def _pack_rows(lines: list[str], prefix: str, header: str, metadata: dict, row_budget: int, max_rows: int):
    """Keep ordinary rows intact; put each partial oversized row in its own batch."""
    chunks = []
    pending: list[str] = []
    low = high = 0

    def flush():
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
        if len(line) > row_budget:
            flush()
            cells = _cells(line)
            for part in _split_row(line, row_budget):
                chunks.append(
                    Document(
                        page_content=prefix + header + "\n" + part,
                        metadata={
                            **metadata,
                            "table_row_start": number,
                            "table_row_end": number,
                            "table_row_key": cells[0] if cells else "",
                        },
                    )
                )
            continue
        if pending and (len(pending) >= max_rows or len("\n".join([*pending, line])) > row_budget):
            flush()
        if not pending:
            low = number
        high = number
        pending.append(line)
    flush()
    return chunks
