"""Shared parser and splitter contracts for uploads and storage ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from domain.knowledge.enums import DocDomain


if TYPE_CHECKING:
    from pathlib import Path

    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.protocol import DocumentProfile
    from domain.ingestion.raw_document import RawDocument


@dataclass(frozen=True)
class FileMeta:
    """Transport metadata known to the caller (S3 key, filename, etc.)."""

    source_key: str
    filename: str
    extension: str
    size_bytes: int


@dataclass(frozen=True)
class SplitContext:
    """Domain-dependent splitting parameters (domain types in a port are acceptable)."""

    domain: str = DocDomain.GENERAL.value
    profile: DocumentProfile | None = None
    settings: DomainSettingsPort | None = None
    legal_mode: bool = False


@runtime_checkable
class DocumentParserPort(Protocol):
    """Parses a file into domain RawDocuments.

    The adapter handles format-specific logic (format handler registry, merge_pdf_pages,
    tuple convention) and returns domain-typed documents. Callers own
    workflow-specific quality assessment and error reporting.
    """

    def supports(self, extension: str) -> bool: ...
    def parse(self, path: Path, meta: FileMeta) -> list[RawDocument]: ...


@runtime_checkable
class DocumentSplitterPort(Protocol):
    """Splits parsed documents into chunks for embedding.

    The adapter encapsulates splitting strategy selection (general, legal,
    domain-profile-based).
    """

    def split(self, documents: list[RawDocument], context: SplitContext) -> list[RawDocument]: ...
