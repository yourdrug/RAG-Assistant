from __future__ import annotations

from pathlib import Path

from langchain.schema import Document

from infrastructure.ml.ingestion import (
    PARSERS,
    parse_docx_sections,
    parse_markdown_sections,
    parse_pdf,
)
from infrastructure.ml.ingestion import split_documents as _split_documents


class LangchainDocumentParser:
    """Parses files into LangChain Documents using infrastructure parsers."""

    def parse(self, file_path: Path) -> list[Document]:
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return parse_pdf(file_path)

        if ext == ".md":
            return self._sections_to_documents(parse_markdown_sections(file_path), file_path)

        if ext in (".docx", ".doc"):
            return self._sections_to_documents(parse_docx_sections(file_path), file_path)

        parser = PARSERS.get(ext)
        if parser is None:
            raise RuntimeError(f"Unsupported format: {ext}")

        text = parser(file_path)
        if not text or len(text.strip()) < 20:
            raise RuntimeError("Too little text in document")

        return [Document(page_content=text, metadata={"source": file_path.name})]

    @staticmethod
    def _sections_to_documents(sections: list[tuple[str | None, str]], file_path: Path) -> list[Document]:
        docs = []
        for heading, content in sections:
            if not content.strip():
                continue
            metadata: dict = {"source": file_path.name}
            if heading:
                metadata["section"] = heading
            docs.append(Document(page_content=content, metadata=metadata))

        if not docs:
            raise RuntimeError("Too little text in document")
        return docs


class LangchainDocumentSplitter:
    """Splits documents into chunks using LangChain text splitters."""

    def split(self, documents: list[Document]) -> list[Document]:
        return _split_documents(documents)
