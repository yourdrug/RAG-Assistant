"""Adapter bridging TextQualityAssessorPort to the pdf_diag heuristic."""

from __future__ import annotations

from infrastructure.ml.pdf_diag import is_garbled


class TextQualityAssessorAdapter:
    """Implements TextQualityAssessorPort using the existing pdf_diag heuristic."""

    def is_garbled(self, text: str) -> bool:
        return is_garbled(text)
