"""Compatibility exports; use application.documents."""

from application.documents.persistence import (
    persist_document_result,
)


__all__ = ["persist_document_result"]
