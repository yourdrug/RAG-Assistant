"""Chunk-related DTOs -- immutable data-transfer objects for chunk operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkItemDTO:
    id: int
    document_id: int
    chunk_index: int
    content: str
    filename: str
    visibility: str
    doc_domain: str
    owner_id: int | None
    group_id: int | None
    edited_at: str | None
    edited_by: int | None
    manual: bool
    creation_date: str | None
    content_hash: str | None
    section: str | None = None
    heading: str | None = None
    heading_level: int | None = None
    content_type: str | None = None
    doc_title: str | None = None
    doc_type: str | None = None


@dataclass(frozen=True)
class EditChunkResult:
    id: int
    document_id: int
    chunk_index: int
    content: str
    edited_at: str
    edited_by: int
    manual: bool
    warning: str | None = None


@dataclass(frozen=True)
class AddChunkResult:
    id: int
    document_id: int
    chunk_index: int
    content: str
    manual: bool
    warning: str | None = None


@dataclass(frozen=True, kw_only=True)
class ChunkCursorQuery:
    limit: int = 50
    cursor: str | None = None
    direction: str = "next"
    content_hashes: list[str] | None = None


@dataclass(frozen=True)
class IndexedPointSnapshot:
    """What the vector index holds for one point; vector values are never exposed."""

    metadata: dict[str, Any]
    page_content: str | None
    dense_dim: int | None
    sparse_terms: int | None


@dataclass(frozen=True)
class IndexFieldComparison:
    name: str
    expected: str | int | None
    actual: str | int | None
    matches: bool


@dataclass(frozen=True)
class ChunkIndexStateDTO:
    chunk_id: int
    document_id: int
    document_status: str
    indexed: bool
    in_sync: bool
    dense_dim: int | None = None
    sparse_terms: int | None = None
    embedding_model: str | None = None
    content_matches: bool | None = None
    fields: list[IndexFieldComparison] = field(default_factory=list)
    payload_metadata: dict[str, Any] = field(default_factory=dict)
