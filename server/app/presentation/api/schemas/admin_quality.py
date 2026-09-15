"""Admin quality & diagnostics schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentQualityItem(BaseModel):
    id: int
    filename: str
    status: str
    quality_score: float | None = None
    warning_message: str | None = None
    chunks: int | None = None
    chars: int | None = None
    indexed_at: datetime | None = None


class DocumentQualityListResponse(BaseModel):
    documents: list[DocumentQualityItem]
    total: int


class PageDiagnostic(BaseModel):
    page: int
    type: str
    chars: int
    description: str


class DocumentDiagnoseResponse(BaseModel):
    document_id: int
    filename: str
    total_pages: int
    pages: list[PageDiagnostic]
    summary: dict


class DryRunPageResult(BaseModel):
    page: int
    type: str
    content_type: str
    chars: int
    preview: str
    full_text: str = ""
    problem_spans: list[tuple[int, int]] = Field(default_factory=list)
    previous_type: str | None = None
    image_available: bool = False
    unit_kind: str = "page"
    label: str = ""


class DryRunResponse(BaseModel):
    filename: str
    total_pages: int
    pages: list[DryRunPageResult]
    total_chars: int
    quality_score: float
    warning: str | None = None
    full_text_preview: str
    summary: dict
    suggestion: str | None = None
    preview_id: str | None = None


class PageImageResponse(BaseModel):
    image_base64: str
    page: int
