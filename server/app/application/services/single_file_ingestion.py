"""Compatibility exports; use application.ingestion."""

from application.ingestion.single_file import (
    SingleFileIngestionWorkflow,
)


__all__ = ["SingleFileIngestionWorkflow"]
