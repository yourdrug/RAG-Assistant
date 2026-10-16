"""Compatibility exports; use application.ingestion."""

from application.ingestion.loader import (
    S3DocumentLoader,
)


__all__ = ["S3DocumentLoader"]
