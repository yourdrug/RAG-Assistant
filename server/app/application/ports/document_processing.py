"""Document processing ports — abstract interfaces for infrastructure services.

Lives in the application layer so that DocumentProcessor depends on ports,
not on concrete implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ContentExtractorPort(Protocol):
    """Extracts structured metadata from document content."""

    def extract_article_number(self, text: str) -> str | None: ...
    def extract_date_from_filename(self, filename: str) -> str | None: ...


@runtime_checkable
class PDFQualityAssessorPort(Protocol):
    """Assesses the quality of PDF text extraction."""

    def assess(self, pdf_path: Path, documents: list) -> object: ...


@runtime_checkable
class MetricsCollectorPort(Protocol):
    """Collects processing metrics (Prometheus counters/histograms)."""

    def inc_chunks(self, count: int) -> None: ...
    def inc_documents(self, status: str) -> None: ...
    def observe_duration(self, status: str, seconds: float) -> None: ...
    def observe_pdf_pages(self, quality: str, count: int) -> None: ...
    def observe_pdf_bad_ratio(self, ratio: float) -> None: ...
