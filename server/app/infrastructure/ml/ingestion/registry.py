"""Format handlers shared by ingestion and preview dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from infrastructure.ml.ingestion.docx import read_docx
from infrastructure.ml.ingestion.markdown import extract_doc_title, parse_markdown_sections
from infrastructure.ml.ingestion.pdf import parse_pdf
from infrastructure.ml.ingestion.rtf import RtfDocumentParser
from infrastructure.ml.ingestion.sections import ParsedFormat, Section
from infrastructure.ml.ingestion.txt import parse_txt
from infrastructure.ml.preview.docx_preview_strategy import DocxPreviewStrategy
from infrastructure.ml.preview.pdf_preview_strategy import PdfPreviewStrategy
from infrastructure.ml.preview.rtf_preview_strategy import RtfPreviewStrategy

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from application.documents.diagnostics import PDFDiagnosticService

    from application.documents import DocumentPreviewStrategy
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.registry import DomainProfileRegistry


MIN_DOCUMENT_CHARS = 20


@dataclass(frozen=True)
class FormatContext:
    domain_registry: DomainProfileRegistry | None = None
    domain_settings: DomainSettingsPort | None = None
    diag_service: object | None = None


@dataclass(frozen=True)
class FormatHandler:
    extensions: tuple[str, ...]
    parse: Callable[[Path, FormatContext], ParsedFormat]
    preview: Callable[[FormatContext], DocumentPreviewStrategy] | None = None


def docx_sections(path: Path, context: FormatContext) -> ParsedFormat:
    del context
    _text, sections, metadata = read_docx(path)
    return ParsedFormat(sections, metadata)


def markdown_sections(path: Path, context: FormatContext) -> ParsedFormat:
    del context
    title = extract_doc_title(path)
    return ParsedFormat(parse_markdown_sections(path), {"doc_title": title} if title else {})


def pdf_sections(path: Path, context: FormatContext) -> ParsedFormat:
    del context
    return ParsedFormat(
        [Section(None, doc.page_content, metadata=dict(doc.metadata)) for doc in parse_pdf(path)],
        structured=False,
    )


def rtf_sections(path: Path, context: FormatContext) -> ParsedFormat:
    return RtfDocumentParser(context.domain_registry, context.domain_settings).parse(path)


def txt_sections(path: Path, context: FormatContext) -> ParsedFormat:
    del context
    text, _metadata = parse_txt(path)

    if not text or len(text.strip()) < MIN_DOCUMENT_CHARS:
        message = "Too little text in document"
        raise RuntimeError(message)

    return ParsedFormat([Section(None, text)], structured=False)


def pdf_preview(context: FormatContext) -> DocumentPreviewStrategy:
    return PdfPreviewStrategy(cast("PDFDiagnosticService", context.diag_service))


def docx_preview(context: FormatContext) -> DocumentPreviewStrategy:
    del context
    return DocxPreviewStrategy()


def rtf_preview(context: FormatContext) -> DocumentPreviewStrategy:
    return RtfPreviewStrategy(context.domain_registry, context.domain_settings)


FORMAT_HANDLERS = {
    extension: handler
    for handler in (
        FormatHandler((".pdf",), pdf_sections, pdf_preview),
        FormatHandler((".docx",), docx_sections, docx_preview),
        FormatHandler((".rtf",), rtf_sections, rtf_preview),
        FormatHandler((".md",), markdown_sections),
        FormatHandler((".txt",), txt_sections),
    )
    for extension in handler.extensions
}
