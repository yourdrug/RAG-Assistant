"""Content extraction adapter — wraps infrastructure ML functions behind ports."""

from __future__ import annotations

from pathlib import Path

from application.ports.document_processing import PDFQualityReport
from infrastructure.ml.ingestion import extract_article_number, extract_date_from_filename
from infrastructure.ml.pdf_diag import assess_pdf_extraction_quality


class MLContentExtractor:
    """Adapts ML extraction functions behind ContentExtractorPort."""

    def extract_article_number(self, text: str) -> str | None:
        return extract_article_number(text)

    def extract_date_from_filename(self, filename: str) -> str | None:
        return extract_date_from_filename(filename)


class MLPDFQualityAssessor:
    """Adapts PDF quality assessment behind PDFQualityAssessorPort."""

    def assess(self, pdf_path: Path, documents: list) -> PDFQualityReport:
        r = assess_pdf_extraction_quality(pdf_path, documents)
        return PDFQualityReport(
            total_pages=r.total_pages,
            n_ok=r.n_ok,
            n_missing=r.n_missing,
            n_garbled=r.n_garbled,
            bad_ratio=r.bad_ratio,
        )
