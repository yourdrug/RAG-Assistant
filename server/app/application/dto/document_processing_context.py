"""Compatibility exports; use application.documents."""

from application.documents.processing_context import (
    ProcessingContext,
    empty_versioning_result,
)


__all__ = ["ProcessingContext", "empty_versioning_result"]
