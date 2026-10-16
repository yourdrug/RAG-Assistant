"""Compatibility exports; use application.documents."""

from application.documents.quality import (
    QualityOutcome,
    assess_document_quality,
)


__all__ = ["QualityOutcome", "assess_document_quality"]
