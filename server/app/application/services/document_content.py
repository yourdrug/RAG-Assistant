"""Parse, assess and split document content in bounded worker threads."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

from application.dto.document_processing_context import ProcessingContext
from application.ports.document_processing import (
    ContentExtractorPort,
    MetricsCollectorPort,
    PDFQualityAssessorPort,
    TextQualityAssessorPort,
)
from application.services.document_quality import assess_document_quality
from domain.entities.raw_document import RawDocument
from domain.services.document_parser import DocumentParser, DocumentSplitter

T = TypeVar("T")


class DocumentContentProcessor:
    """Own the parser/splitter dependencies and their shared concurrency limit."""

    def __init__(
        self,
        parser: DocumentParser,
        splitter: DocumentSplitter,
        extractor: ContentExtractorPort,
        pdf_assessor: PDFQualityAssessorPort,
        text_assessor: TextQualityAssessorPort,
        metrics: MetricsCollectorPort,
    ) -> None:
        self._parser = parser
        self._splitter = splitter
        self._extractor = extractor
        self._pdf_assessor = pdf_assessor
        self._text_assessor = text_assessor
        self._metrics = metrics
        self._blocking_semaphore = asyncio.Semaphore(1)

    async def _offload(self, function: Callable[..., T], *args, **kwargs) -> T:
        async with self._blocking_semaphore:
            return await asyncio.to_thread(function, *args, **kwargs)

    async def parse(self, ctx: ProcessingContext) -> None:
        """Extract text and assess quality before classification or replacement."""
        if ctx.temp_path is None:
            raise RuntimeError("Document content requires a downloaded temporary file")
        ctx.docs = await self._offload(self._parser.parse, ctx.temp_path)
        if not ctx.docs:
            raise RuntimeError(
                "Текст не извлечён — документ похож на скан, и OCR не смог распознать содержимое."
            )
        outcome = await self._offload(
            assess_document_quality,
            ctx.temp_path,
            ctx.original_filename,
            ctx.document_id,
            ctx.docs,
            pdf_assessor=self._pdf_assessor,
            text_quality_assessor=self._text_assessor,
            metrics=self._metrics,
        )
        ctx.quality = outcome.report
        if outcome.warning:
            ctx.warnings.append(outcome.warning)

    async def split(self, ctx: ProcessingContext) -> list[RawDocument]:
        """Attach source/date metadata, split and add section prefixes."""
        if ctx.doc_domain is None:
            raise RuntimeError(f"domain classification produced no domain for doc {ctx.document_id}")
        ctx.raw_chunks = await self._offload(
            self._attach_metadata_and_split, ctx.docs, ctx.original_filename, ctx.doc_domain
        )
        return ctx.raw_chunks

    def _attach_metadata_and_split(
        self, docs: list[RawDocument], filename: str, domain: str
    ) -> list[RawDocument]:
        doc_date = self._extractor.extract_date_from_filename(filename)
        for doc in docs:
            doc.metadata["source"] = filename
            if doc_date:
                doc.metadata["doc_date"] = doc_date
        chunks = self._splitter.split(docs, domain=domain)
        for chunk in chunks:
            section = chunk.metadata.get("section")
            if section:
                chunk.page_content = f"[Раздел: {section}]\n{chunk.page_content}"
        return chunks
