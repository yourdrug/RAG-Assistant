"""Document processing ports — abstract interfaces for infrastructure services.

Lives in the application layer so that DocumentProcessor depends on ports,
not on concrete implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

# Re-export from domain for backward compatibility.
from domain.value_objects.pdf_quality_report import PDFQualityReport  # noqa: F401


@runtime_checkable
class ContentExtractorPort(Protocol):
    """Extracts structured metadata from document content."""

    def extract_date_from_filename(self, filename: str) -> str | None: ...


@runtime_checkable
class PDFQualityAssessorPort(Protocol):
    """Assesses the quality of PDF text extraction."""

    def assess(self, pdf_path: Path, documents: list) -> PDFQualityReport: ...


@runtime_checkable
class TextQualityAssessorPort(Protocol):
    """Assesses whether extracted text is garbled / low-quality."""

    def is_garbled(self, text: str) -> bool: ...


@runtime_checkable
class MetricsCollectorPort(Protocol):
    """Collects processing metrics (Prometheus counters/histograms)."""

    def inc_chunks(self, count: int) -> None: ...
    def inc_documents(self, status: str) -> None: ...
    def observe_duration(self, status: str, seconds: float) -> None: ...
    def observe_pdf_pages(self, quality: str, count: int) -> None: ...
    def observe_pdf_bad_ratio(self, ratio: float) -> None: ...
    def observe_domain_classification(self, domain: str, level: str, confidence: float) -> None: ...
    def inc_domain_ambiguous(self, candidates: str) -> None: ...
