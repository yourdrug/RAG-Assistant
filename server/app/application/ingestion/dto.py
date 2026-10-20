"""Ingestion-related DTOs -- immutable data-transfer objects for the ingestion API."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict


if TYPE_CHECKING:
    from domain.knowledge.enums import DocumentVisibility


from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NotRequired


if TYPE_CHECKING:
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
    ingestion_key: str = ""


@dataclass(frozen=True)
class IngestRegistryResult:
    total_files: int
    total_chunks: int
    files: list[IngestRegistryItemDTO] = field(default_factory=list)


class RegistryRecord(TypedDict):
    """Registry snapshot shared by ingestion workflows."""

    hash: str
    source: str
    chunks: int
    chars: int
    indexed_at: NotRequired[datetime | None]
    filename: NotRequired[str]


class SyncOneDocumentOptions(TypedDict):
    full_text: str
    visibility: DocumentVisibility
    owner_id: int | None
    group_id: int | None
