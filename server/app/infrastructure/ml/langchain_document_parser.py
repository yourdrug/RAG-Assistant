"""LangChain Document Parser — infrastructure adapter for domain.services.document_parser."""

from __future__ import annotations

from pathlib import Path

from langchain.schema import Document

from infrastructure.ml.ingestion import PARSERS, parse_pdf
from infrastructure.ml.ingestion import split_documents as _split_documents


class LangchainDocumentParser:
    """Parses files into LangChain Documents using infrastructure parsers."""

    def parse(self, file_path: Path) -> list[Document]:
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return parse_pdf(file_path)

        parser = PARSERS.get(ext)
        if parser is None:
            raise RuntimeError(f"Unsupported format: {ext}")

        result = parser(file_path)
        # Handle both old (str) and new (tuple) return types
        if isinstance(result, tuple):
            text, extra_meta = result
        else:
            text, extra_meta = result, {}

        if not text or len(text.strip()) < 20:
            raise RuntimeError("Too little text in document")

        metadata = {"source": file_path.name}
        # Add section_count as a proxy for page numbers in non-PDF formats
        section_count = extra_meta.get("section_count")
        if section_count and section_count > 1:
            metadata["section_count"] = section_count

        return [Document(page_content=text, metadata=metadata)]


class LangchainDocumentSplitter:
    """Splits documents into chunks using LangChain text splitters."""

    def split(self, documents: list[Document]) -> list[Document]:
        return _split_documents(documents)
