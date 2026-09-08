"""PDF diagnostic DTOs -- data transfer objects for dry-run preview results."""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.value_objects.page_content_type import PageContentType
from domain.value_objects.preview_unit_kind import PreviewUnitKind


@dataclass(frozen=True)
class DryRunPageResult:
    page: int
    type: str
    content_type: str = PageContentType.TEXT
    chars: int = 0
    preview: str = ""
    full_text: str = ""
    problem_spans: list[tuple[int, int]] = field(default_factory=list)
    previous_type: str | None = None
    unit_kind: str = PreviewUnitKind.PAGE
    label: str = ""


@dataclass(frozen=True)
class DryRunResult:
    filename: str
    total_pages: int = 0
    pages: list[DryRunPageResult] = field(default_factory=list)
    total_chars: int = 0
    quality_score: float = 0.0
    warning: str | None = None
    full_text_preview: str = ""
    summary: dict[str, int] = field(default_factory=dict)
    suggestion: str | None = None
    preview_id: str | None = None
