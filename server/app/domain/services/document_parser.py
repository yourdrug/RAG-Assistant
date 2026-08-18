"""Document parser protocol -- abstracts file parsing and text splitting for the ML pipeline.

Implementations live in ``infrastructure.ml.ingestion`` and
``infrastructure.ml.langchain_document_parser``; the domain service
(``DocumentProcessor``) depends only on this protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from domain.entities.raw_document import RawDocument
from domain.value_objects.doc_domain import DocDomain


class DocumentParser(Protocol):
    """Parses a file into domain RawDocuments."""

    def parse(self, file_path: Path) -> list[RawDocument]: ...


class DocumentSplitter(Protocol):
    """Splits documents into chunks for embedding."""

    def split(
        self, documents: list[RawDocument], domain: str = DocDomain.GENERAL.value
    ) -> list[RawDocument]: ...
