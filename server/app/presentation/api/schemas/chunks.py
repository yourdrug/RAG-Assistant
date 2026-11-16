"""Chunk management schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from domain.knowledge.enums import DocDomain
from presentation.api.schemas.documents import DocumentResponse


class ChunkResponse(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    content: str
    filename: str = ""
    visibility: str = ""
    doc_domain: str = DocDomain.GENERAL.value
    owner_id: int | None = None
    group_id: int | None = None
    edited_at: str | None = None
    edited_by: int | None = None
    manual: bool = False
    creation_date: str | None = None
    warning: str | None = None
    content_hash: str | None = None
    section: str | None = None
    heading: str | None = None
    heading_level: int | None = None
    content_type: str | None = None
    doc_title: str | None = None
    doc_type: str | None = None
    char_count: int | None = None
    total_chunks: int | None = None


class ChunkCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1, max_length=10000)
    page: int | None = Field(None, description="Page number (optional)")
    section: str | None = Field(None, description="Section name (optional)")


class ChunkEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1, max_length=10000)


class ChunkListResponse(BaseModel):
    chunks: list[ChunkResponse]
    total: int
    document_id: int


class ChunkCursorListResponse(BaseModel):
    chunks: list[ChunkResponse]
    next_cursor: str | None = None
    prev_cursor: str | None = None
    document_id: int


class ChunkDeleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    chunk_id: int


__all__ = [
    "ChunkCreateRequest",
    "ChunkCursorListResponse",
    "ChunkDeleteResponse",
    "ChunkEditRequest",
    "ChunkListResponse",
    "ChunkResponse",
    "DocumentResponse",
]


class IndexFieldComparisonResponse(BaseModel):
    name: str
    expected: str | int | None = None
    actual: str | int | None = None
    matches: bool


class ChunkIndexStateResponse(BaseModel):
    """Admin diagnostics: the chunk's point in the vector index versus Postgres."""

    chunk_id: int
    document_id: int
    document_status: str
    indexed: bool
    in_sync: bool
    dense_dim: int | None = None
    sparse_terms: int | None = None
    embedding_model: str | None = None
    content_matches: bool | None = None
    fields: list[IndexFieldComparisonResponse] = Field(default_factory=list)
    payload_metadata: dict[str, Any] = Field(default_factory=dict)
