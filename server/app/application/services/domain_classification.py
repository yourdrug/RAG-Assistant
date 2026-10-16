"""Compatibility exports; use application.documents."""

from application.documents.classification import (
    DomainClassification,
    classify_document_text,
    classify_processing_document,
)


__all__ = ["DomainClassification", "classify_document_text", "classify_processing_document"]
