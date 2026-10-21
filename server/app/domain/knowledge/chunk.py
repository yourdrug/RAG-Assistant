"""Knowledge domain: chunk."""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from domain.knowledge.enums import DocDomain
from domain.shared.exceptions import ValidationError


if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import date, datetime


@dataclass
class Chunk:
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


TABLE_CONTEXT_MAX_CHUNKS = 32


CHUNK_CONTEXT_FIELDS = frozenset(
    {
        "section",
        "heading",
        "heading_level",
        "parent_section",
        "doc_title",
        "doc_type",
        "doc_date",
        "content_type",
        "page",
        "page_start",
        "page_end",
        "pages",
        "unit_kind",
        "unit_heading_text",
        "parent_units",
        "article_number",
        "chapter_number",
        "section_number",
        "clause_number",
        "point_number",
        "subpoint_number",
        "subpoint_num_number",
        "step_number",
        "step_numbers",
        "table_id",
        "table_header",
        "table_row_start",
        "table_row_end",
        "table_row_key",
        "domain_metadata",
    }
)


def extract_chunk_context(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Copy interpretation fields, excluding identity, ACL and temporal state."""
    return {key: deepcopy(value) for key, value in metadata.items() if key in CHUNK_CONTEXT_FIELDS}


@dataclass(frozen=True)
class ChunkInsert:
    """Content and interpretation fields of one chunk in a document replacement."""

    content: str
    content_hash: str | None = None
    section: str | None = None
    heading: str | None = None
    heading_level: int | None = None
    content_type: str | None = None
    doc_title: str | None = None
    doc_type: str | None = None
    context_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkRecord:
    """A single chunk match from substring search."""

    chunk_id: int
    document_id: int
    filename: str
    content: str
    chunk_index: int
    visibility: str = ""
    doc_domain: str = DocDomain.GENERAL.value
    owner_id: int | None = None
    group_id: int | None = None
    edited_at: datetime | None = None
    edited_by: int | None = None
    manual: bool = False
    creation_date: datetime | None = None
    content_hash: str | None = None
    section: str | None = None
    heading: str | None = None
    heading_level: int | None = None
    content_type: str | None = None
    doc_title: str | None = None
    doc_type: str | None = None
    act_version_id: int | None = None
    act_id: int | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    is_current: bool = True
    context_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkStats:
    """Aggregate statistics for chunks of a document."""

    total_chunks: int
    total_chars: int


def encode_cursor(chunk_index: int, chunk_id: int) -> str:
    """Encode a (chunk_index, chunk_id) pair into an opaque base64url cursor."""
    payload = f"{chunk_index}:{chunk_id}".encode()
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[int, int]:
    """Decode an opaque cursor back to ``(chunk_index, chunk_id)``.

    Raises ``ValidationError`` if the cursor is malformed.
    """
    try:
        # Re-pad to a multiple of 4 for urlsafe_b64decode
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded).decode("utf-8")
        return parse_cursor_parts(raw)
    except Exception as exc:
        msg = f"Invalid cursor: {exc}"
        raise ValidationError(msg) from exc


def parse_cursor_parts(raw: str) -> tuple[int, int]:
    parts = raw.split(":")
    pair_length = 2

    if len(parts) != pair_length:
        msg = "expected exactly two parts"
        raise ValueError(msg)

    return int(parts[0]), int(parts[1])


SENTENCE_UNIT_KIND = "sentence"
