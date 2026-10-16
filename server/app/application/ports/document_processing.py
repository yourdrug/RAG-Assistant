"""Compatibility exports; use application.documents."""

from application.documents.processing_ports import (
    ContentExtractorPort,
    MetricsCollectorPort,
    PDFQualityAssessorPort,
    TextQualityAssessorPort,
)


__all__ = [
    "ContentExtractorPort",
    "MetricsCollectorPort",
    "PDFQualityAssessorPort",
    "TextQualityAssessorPort",
]
