"""PDF quality report -- pure data structure for extraction quality assessment."""

from __future__ import annotations

from dataclasses import dataclass


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
        return self.total_pages > 0 and self.bad_ratio > 0.3
