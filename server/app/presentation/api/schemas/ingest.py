"""Ingest schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class IngestStatusResponse(BaseModel):
    job_id: int | None = None
    status: str
    mode: str | None = None
    docs_dir: str | None = None
    file: str | None = None
    force: bool | None = None


class IngestRegistryItem(BaseModel):
    filename: str
    ingestion_key: str = ""
    chunks: int | None = None
    chars: int | None = None
    indexed_at: datetime | str | None = None
    source: str | None = None


class IngestRegistryResponse(BaseModel):
    total_files: int
    total_chunks: int
    files: list[IngestRegistryItem]
