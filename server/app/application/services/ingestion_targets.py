"""Compatibility exports; use application.ingestion."""

from application.ingestion.targets import (
    S3IngestionTargets,
    S3UploadService,
)


__all__ = ["S3IngestionTargets", "S3UploadService"]
