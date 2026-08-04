"""Document Parser Protocol — abstracts file parsing for the ML pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from domain.entities.raw_document import RawDocument


class DocumentParser(Protocol):
    """Parses a file into domain RawDocuments."""

    def parse(self, file_path: Path) -> list[RawDocument]: ...


class DocumentSplitter(Protocol):
    """Splits documents into chunks for embedding."""

    def split(self, documents: list[RawDocument]) -> list[RawDocument]: ...
