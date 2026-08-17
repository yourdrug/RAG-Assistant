"""LangChain document parser -- bridges RawDocument entities to LangChain Documents.

Wraps the low-level parsers from ``infrastructure.ml.ingestion`` and exposes a
``DocumentParser`` / ``DocumentSplitter`` pair compatible with the domain
service layer (``DocumentProcessor``).
"""

from __future__ import annotations

from pathlib import Path

from domain.entities.raw_document import RawDocument
from langchain.schema import Document

from infrastructure.ml.ingestion import (
    PARSERS,
    _parse_docx,
    parse_docx_sections,
    parse_markdown_sections,
    parse_pdf,
)
from infrastructure.ml.ingestion import split_documents as _split_documents


def _lc_to_raw(docs) -> list[RawDocument]:
    return [RawDocument(page_content=d.page_content, metadata=dict(d.metadata)) for d in docs]


def _raw_to_lc(docs: list[RawDocument]):
    return [Document(page_content=d.page_content, metadata=d.metadata) for d in docs]


class LangchainDocumentParser:
    """Parses files into domain RawDocuments using LangChain infrastructure."""

    def parse(self, file_path: Path) -> list[RawDocument]:
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return _lc_to_raw(parse_pdf(file_path))

        if ext == ".md":
            return self._sections_to_documents(parse_markdown_sections(file_path), file_path)

        if ext in (".docx", ".doc"):
            sections = parse_docx_sections(file_path)
            # Get page metadata from docx page break detection
            _text, page_meta = _parse_docx(file_path)
            return self._sections_to_documents(sections, file_path, page_meta)

        parser = PARSERS.get(ext)
        if parser is None:
            raise RuntimeError(f"Unsupported format: {ext}")

        result = parser(file_path)
        # Parsers may return str or tuple[str, dict]
        if isinstance(result, tuple):
            text, _meta = result
        else:
            text = result
        if not text or len(text.strip()) < 20:
            raise RuntimeError("Too little text in document")

        return [RawDocument(page_content=text, metadata={"source": file_path.name})]

    @staticmethod
    def _sections_to_documents(
        sections: list[tuple[str | None, str]],
        file_path: Path,
        page_meta: dict | None = None,
    ) -> list[RawDocument]:
        docs = []
        for heading, content in sections:
            if not content.strip():
                continue

            metadata: dict = {"source": file_path.name}
            if heading:
                metadata["section"] = heading
            if page_meta:
                metadata.update(page_meta)

            # Handle table blocks tagged by parse_markdown_sections
            if content.startswith("\x00TABLE:"):
                metadata["content_type"] = "table"
                content = content[len("\x00TABLE:"):]

            if not content.strip():
                continue
            docs.append(RawDocument(page_content=content, metadata=metadata))

        if not docs:
            raise RuntimeError("Too little text in document")
        return docs


class LangchainDocumentSplitter:
    """Splits domain RawDocuments into chunks using LangChain text splitters."""

    def split(self, documents: list[RawDocument], domain: str = "general") -> list[RawDocument]:
        return _lc_to_raw(_split_documents(_raw_to_lc(documents), domain=domain))
