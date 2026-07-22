"""Ingestion-related DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IngestCommand:
    docs_dir: str
    reset: bool = False
    prefix: str | None = None


@dataclass(frozen=True)
class IngestSingleFileCommand:
    file_path: str
    force: bool = False


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
    indexed_at: str
    source: str


@dataclass(frozen=True)
class IngestRegistryResult:
    total_files: int
    total_chunks: int
    files: list[IngestRegistryItemDTO] = field(default_factory=list)
