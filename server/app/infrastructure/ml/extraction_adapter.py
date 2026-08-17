"""Content extraction adapter — wraps infrastructure ML functions behind ports."""

from __future__ import annotations

from pathlib import Path

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

    def assess(self, pdf_path: Path, documents: list) -> object:
        return assess_pdf_extraction_quality(pdf_path, documents)
