"""Ingestion-related DTOs -- immutable data-transfer objects for the ingestion API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class IngestStatusResult:
    status: str
    mode: str | None = None
    file: str | None = None
    force: bool | None = None
    docs_dir: str | None = None


@dataclass(frozen=True)
class IngestRegistryItemDTO:
    filename: str
    chunks: int
    chars: int
    indexed_at: datetime | None = None
    source: str = ""


@dataclass(frozen=True)
class IngestRegistryResult:
    total_files: int
    total_chunks: int
    files: list[IngestRegistryItemDTO] = field(default_factory=list)
