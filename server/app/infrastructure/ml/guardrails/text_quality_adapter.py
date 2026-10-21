"""Adapter bridging TextQualityAssessorPort to domain text quality rules."""

from __future__ import annotations

from domain.ingestion.text_quality import is_garbled


class TextQualityAssessorAdapter:
    """Implements TextQualityAssessorPort using the shared text quality heuristic."""

    def is_garbled(self, text: str) -> bool:
        return is_garbled(text)
