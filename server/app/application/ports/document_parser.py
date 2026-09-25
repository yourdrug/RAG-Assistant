"""Document parser and splitter ports — application-layer abstractions for parsing files.

Consolidates ``DocumentParser``/``DocumentSplitter`` from
``domain/services/document_parser.py`` into a single application port module.
The domain module retains re-exports for backward compatibility during the
transition period.

Ports operate on ``domain.entities.raw_document.RawDocument`` and
``domain.domain_profile.protocol.DomainProfile`` — no infrastructure types
leak through the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from domain.entities.raw_document import RawDocument

if TYPE_CHECKING:
    from domain.domain_profile.settings_port import DomainSettingsPort
    from domain.domain_profile.protocol import DomainProfile


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

    domain: str = "general"
    profile: DomainProfile | None = None
    settings: DomainSettingsPort | None = None
    legal_mode: bool = False


@runtime_checkable
class DocumentParserPort(Protocol):
    """Parses a file into domain RawDocuments.

    The adapter handles format-specific logic (PARSERS registry, merge_pdf_pages,
    tuple convention) and returns domain-typed documents.  The caller owns
    quality policies (e.g. "skip if < 20 chars").
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
