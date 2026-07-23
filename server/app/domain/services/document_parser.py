"""Document Parser Protocol — abstracts file parsing for the ML pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from langchain.schema import Document


class DocumentParser(Protocol):
    """Parses a file into LangChain Documents."""

    def parse(self, file_path: Path) -> list[Document]: ...


class DocumentSplitter(Protocol):
    """Splits documents into chunks for embedding."""

    def split(self, documents: list[Document]) -> list[Document]: ...
