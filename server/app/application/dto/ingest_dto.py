"""Compatibility exports; use application.ingestion."""

from application.ingestion.dto import (
    IngestRegistryItemDTO,
    IngestRegistryResult,
    IngestStatusResult,
)


__all__ = ["IngestRegistryItemDTO", "IngestRegistryResult", "IngestStatusResult"]
