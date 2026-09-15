"""Exact substring search schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from domain.value_objects.search_mode import SearchMode


class ExactSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=3, max_length=200, description="Search query (min 3 chars)")
    mode: str = Field(
        SearchMode.EXACT.value,
        pattern="^(exact|icontains)$",
        description="exact=pg_trgm ranked, icontains=plain ILIKE",
    )
    limit: int = Field(20, ge=1, le=100)
    document_id: int | None = Field(None, description="Filter by document ID (optional)")


class ExactSearchResult(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    content: str
    chunk_index: int


class ExactSearchResponse(BaseModel):
    query: str
    results: list[ExactSearchResult]
    total: int
