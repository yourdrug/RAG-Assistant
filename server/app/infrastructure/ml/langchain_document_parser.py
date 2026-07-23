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

        text = parser(file_path)
        if not text or len(text.strip()) < 20:
            raise RuntimeError("Too little text in document")

        return [Document(page_content=text, metadata={"source": file_path.name})]


class LangchainDocumentSplitter:
    """Splits documents into chunks using LangChain text splitters."""

    def split(self, documents: list[Document]) -> list[Document]:
        return _split_documents(documents)
