"""Chunk-related DTOs -- immutable data-transfer objects for chunk operations."""

from __future__ import annotations

from dataclasses import dataclass


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
