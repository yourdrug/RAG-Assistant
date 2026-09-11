"""PDF diagnostic adapters — wraps infrastructure PDF/OCR functions."""

from __future__ import annotations

import fitz

from infrastructure.ml.ingestion import clean_pdf_text, ocr_pdf_pages
from infrastructure.ml.pdf_diag import classify_page


class FitzPDFDocument:
    def open(self, path: str):
        return fitz.open(path)

    def get_page_count(self, doc) -> int:
        return len(doc)

    def get_page_text(self, doc, index: int) -> str:
        return doc.load_page(index).get_text("text")

    def find_tables(self, page) -> bool:
        try:
            tables = page.find_tables()
            return bool(tables and tables.tables)
        except Exception:
            return False

    def render_page_image(self, doc, index: int, dpi: int = 100) -> bytes:
        page = doc.load_page(index)
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")

    def close(self, doc) -> None:
        doc.close()


class MLPageClassifier:
    def classify_page(self, text: str, chars: int, page=None) -> tuple[str, str]:
        return classify_page(text, chars, page=page)


class MLTextCleaner:
    def clean(self, text: str) -> str:
        return clean_pdf_text(text)


class MLOcrRunner:
    def ocr_pages(self, pdf, pages: list[int], filename: str) -> dict[int, str]:
        return ocr_pdf_pages(pdf, pages, filename)
