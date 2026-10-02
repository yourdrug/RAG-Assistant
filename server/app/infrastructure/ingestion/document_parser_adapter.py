"""CLI adapters over the same parser and splitter used by API uploads."""

from __future__ import annotations

from pathlib import Path

from application.ports.document_parser import FileMeta, SplitContext
from domain.entities.raw_document import RawDocument
from infrastructure.ml.ingestion.registry import PARSERS
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser, LangchainDocumentSplitter


class IngestionDocumentParser:
    """Add storage identity without changing format-specific parsing rules."""

    def __init__(self, parser: LangchainDocumentParser | None = None) -> None:
        self._parser = parser if parser is not None else LangchainDocumentParser()

    def supports(self, extension: str) -> bool:
        return extension.lower() in PARSERS or extension.lower() == ".pdf"

    def parse(self, path: Path, meta: FileMeta) -> list[RawDocument]:
        if not self.supports(meta.extension):
            return []
        docs = self._parser.parse(path, filename=meta.filename)
        base = {
            "source": meta.source_key,
            "filename": meta.filename,
            "extension": meta.extension,
            "size_bytes": meta.size_bytes,
        }
        return [RawDocument(doc.page_content, {**doc.metadata, **base}) for doc in docs]


class IngestionDocumentSplitter:
    """Use the API splitter, including its context, limits and page handling."""

    def __init__(self, splitter: LangchainDocumentSplitter | None = None) -> None:
        self._splitter = splitter if splitter is not None else LangchainDocumentSplitter()

    def split(self, documents: list[RawDocument], context: SplitContext) -> list[RawDocument]:
        return self._splitter.split(
            documents, domain=context.domain, profile=context.profile, domain_settings=context.settings
        )
