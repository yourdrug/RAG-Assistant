"""Compatibility exports; use application.ingestion."""

from application.ingestion.sync import (
    DocumentSyncService,
)


__all__ = ["DocumentSyncService"]
