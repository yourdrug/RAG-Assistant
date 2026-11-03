"""PDF diagnostic adapters — wraps infrastructure PDF/OCR functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import fitz

from domain.ingestion.text_quality import classify_content
from infrastructure.ml.ingestion import clean_pdf_text, ocr_pdf_pages


if TYPE_CHECKING:
    from application.documents.ports.diagnostics import PDFDocumentHandle


logger = logging.getLogger(__name__)


def page_has_tables(page: object) -> bool:
    try:
        tables = cast("fitz.Page", page).find_tables()
        return bool(tables.tables)
    except Exception:
        logger.debug("PDF table detection failed", exc_info=True)
        return False


def classify_page(text: str, chars: int, page: object | None = None) -> tuple[str, str]:
    ptype, description = classify_content(text, chars, has_table=page is not None and page_has_tables(page))
    return ptype.value, description


class FitzPDFDocument:
    def open(self, path: str) -> fitz.Document:
        return fitz.Document(path)

    def get_page_count(self, doc: PDFDocumentHandle) -> int:
        return len(cast("fitz.Document", doc))

    def get_page_text(self, doc: PDFDocumentHandle, index: int) -> str:
        return cast("fitz.Document", doc).load_page(index).get_text("text")

    def find_tables(self, page: object) -> bool:
        return page_has_tables(page)

    def render_page_image(self, doc: PDFDocumentHandle, index: int, dpi: int = 100) -> bytes:
        page = cast("fitz.Document", doc).load_page(index)
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")

    def close(self, doc: PDFDocumentHandle) -> None:
        cast("fitz.Document", doc).close()


class MLPageClassifier:
    def classify_page(self, text: str, chars: int, page: object | None = None) -> tuple[str, str]:
        return classify_page(text, chars, page=page)


class MLTextCleaner:
    def clean(self, text: str) -> str:
        return clean_pdf_text(text)


class MLOcrRunner:
    def ocr_pages(self, pdf: PDFDocumentHandle, pages: list[int], filename: str) -> dict[int, str]:
        return ocr_pdf_pages(pdf, pages, filename)
