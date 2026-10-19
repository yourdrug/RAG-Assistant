"""Parse, assess and split document content in bounded worker threads."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from application.documents.quality import assess_document_quality
from application.ingestion import FileMeta, SplitContext


if TYPE_CHECKING:
    from collections.abc import Callable

    from application.documents.ports.processing import (
        ContentExtractorPort,
        MetricsCollectorPort,
        PDFQualityAssessorPort,
        TextQualityAssessorPort,
    )
    from application.ingestion import DocumentParserPort, DocumentSplitterPort
    from domain.entities.raw_document import RawDocument
    from domain.value_objects.pdf_quality_report import PDFQualityReport


T = TypeVar("T")
P = ParamSpec("P")


@dataclass(frozen=True)
class ParsedDocumentContent:
    documents: list[RawDocument]
    quality: PDFQualityReport | None
    warning: str | None


class DocumentContentProcessor:
    """Own the parser/splitter dependencies and their shared concurrency limit."""

    def __init__(
        self,
        parser: DocumentParserPort,
        splitter: DocumentSplitterPort,
        extractor: ContentExtractorPort,
        pdf_assessor: PDFQualityAssessorPort,
        *,
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

    async def offload(self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        async with self._blocking_semaphore:
            return await asyncio.to_thread(function, *args, **kwargs)

    async def parse(self, path: Path, filename: str, document_id: int) -> ParsedDocumentContent:
        documents = await self.offload(self.parse_file, path, filename)

        if not documents:
            msg = "Текст не извлечён — документ похож на скан, и OCR не смог распознать содержимое."
            raise RuntimeError(msg)

        outcome = await self.offload(
            assess_document_quality,
            path,
            filename,
            document_id,
            documents,
            pdf_assessor=self._pdf_assessor,
            text_quality_assessor=self._text_assessor,
            metrics=self._metrics,
        )
        return ParsedDocumentContent(documents, outcome.report, outcome.warning)

    def parse_file(self, path: Path, filename: str) -> list[RawDocument]:
        meta = FileMeta(
            source_key=filename,
            filename=filename,
            extension=Path(filename).suffix.lower(),
            size_bytes=path.stat().st_size,
        )
        return self._parser.parse(path, meta)

    async def split(self, documents: list[RawDocument], filename: str, domain: str) -> list[RawDocument]:
        return await self.offload(self.attach_metadata_and_split, documents, filename, domain)

    def attach_metadata_and_split(
        self, docs: list[RawDocument], filename: str, domain: str
    ) -> list[RawDocument]:
        doc_date = self._extractor.extract_date_from_filename(filename)

        for doc in docs:
            doc.metadata["source"] = filename

            if doc_date:
                doc.metadata["doc_date"] = doc_date

        return self._splitter.split(docs, SplitContext(domain=domain))
