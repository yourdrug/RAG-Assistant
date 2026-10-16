"""Compatibility exports; use application.ingestion."""

from application.ingestion.registry import (
    IngestionRegistry,
    s3_file_hash,
)


__all__ = ["IngestionRegistry", "s3_file_hash"]
