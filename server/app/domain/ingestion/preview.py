"""Document preview and extraction quality contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from domain.knowledge.enums import PageContentType


LOW_QUALITY_BAD_RATIO = 0.3


class PreviewUnitKind(StrEnum):
    PAGE = "page"
    SECTION = "section"
    DOCUMENT = "document"


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


@dataclass(frozen=True)
class PDFQualityReport:
    """Result of PDF text extraction quality assessment."""

    total_pages: int
    n_ok: int
    n_missing: int
    n_garbled: int
    bad_ratio: float

    @property
    def is_low_quality(self) -> bool:
        return self.total_pages > 0 and self.bad_ratio > LOW_QUALITY_BAD_RATIO
