"""Compatibility exports; use application.ingestion."""

from application.ingestion.orchestrator import (
    IngestionService,
)


__all__ = ["IngestionService"]
