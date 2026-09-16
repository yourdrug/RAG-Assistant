"""Document parser and splitter adapters — implement ``DocumentParserPort`` / ``DocumentSplitterPort``.

Encapsulates format-specific parsing (PARSERS registry, PDF special-casing,
merge_pdf_pages) and splitting strategy selection (general, legal, domain-profile).
The caller only sees domain ``RawDocument`` types.
"""

from __future__ import annotations

import logging
from pathlib import Path

from application.ports.document_parser import (
    FileMeta,
    SplitContext,
)
from domain.entities.raw_document import RawDocument
from domain.value_objects.doc_domain import DocDomain

log = logging.getLogger("default")


class IngestionDocumentParser:
    """Implements ``DocumentParserPort`` for the ingestion pipeline."""

    def supports(self, extension: str) -> bool:
        from infrastructure.ml.ingestion.registry import PARSERS

        return extension.lower() in PARSERS or extension.lower() == ".pdf"

    def parse(self, path: Path, meta: FileMeta) -> list[RawDocument]:
        ext = meta.extension.lower()

        if ext == ".pdf":
            return self._parse_pdf(path, meta)
        return self._parse_generic(path, meta)

    def _parse_pdf(self, path: Path, meta: FileMeta) -> list[RawDocument]:
        from infrastructure.ml.ingestion import parse_pdf

        pages = parse_pdf(path)
        if not pages:
            return []
        base = self._base_metadata(meta)
        return [
            RawDocument(page_content=doc.page_content, metadata={**base, **doc.metadata}) for doc in pages
        ]

    def _parse_generic(self, path: Path, meta: FileMeta) -> list[RawDocument]:
        from infrastructure.ml.ingestion.registry import PARSERS

        parser_fn = PARSERS.get(meta.extension.lower())
        if parser_fn is None:
            log.debug("SKIP unsupported format: %s", meta.filename)
            return []

        result = parser_fn(path)
        if isinstance(result, tuple):
            text, extra_meta = result
        else:
            text, extra_meta = result, {}

        if not text or len(text.strip()) < 20:
            log.warning("SKIP too little text: %s", meta.filename)
            return []

        base = self._base_metadata(meta)
        base.update(extra_meta)
        return [RawDocument(page_content=text, metadata=base)]

    @staticmethod
    def _base_metadata(meta: FileMeta) -> dict:
        return {
            "source": meta.source_key,
            "filename": meta.filename,
            "extension": meta.extension,
            "size_bytes": meta.size_bytes,
        }


class IngestionDocumentSplitter:
    """Implements ``DocumentSplitterPort`` for the ingestion pipeline."""

    def split(self, documents: list[RawDocument], context: SplitContext) -> list[RawDocument]:
        from infrastructure.ml.ingestion import (
            merge_pdf_pages,
            split_documents,
            split_documents_legal,
        )

        # Convert RawDocument -> langchain Document for splitting
        lc_docs = [
            type("LCDocument", (), {"page_content": d.page_content, "metadata": dict(d.metadata)})()
            for d in documents
        ]
        merged = merge_pdf_pages(lc_docs)

        profile = context.profile
        if (
            profile is not None
            and hasattr(profile, "content_boundaries")
            and profile.content_boundaries()
            and context.settings is not None
        ):
            chunks = split_documents(
                merged, domain=context.domain, profile=profile, settings=context.settings
            )
        elif context.domain == DocDomain.LEGAL.value:
            chunks = split_documents_legal(merged)
        else:
            chunks = split_documents(merged)

        return [RawDocument(page_content=c.page_content, metadata=dict(c.metadata)) for c in chunks]
