"""Admin act version management schemas."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class ActVersionReviewItem(BaseModel):
    id: int
    act_id: int | None
    document_id: int
    document_filename: str
    effective_from: date | None
    effective_to: date | None
    is_current: bool
    date_source: str
    date_confidence: float | None


class ActSummary(BaseModel):
    id: int
    act_type: str
    act_number: str | None
    title: str


class ActVersionListResponse(BaseModel):
    versions: list[ActVersionReviewItem]
    total: int
    acts: list[ActSummary] = []


class ActVersionUpdateRequest(BaseModel):
    effective_from: date | None = None
    effective_to: date | None = None
    act_id: int | None = None
