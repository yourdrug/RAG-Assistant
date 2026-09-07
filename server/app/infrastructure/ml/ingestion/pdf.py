"""PDF parsing — PyMuPDF-based text/table extraction with OCR fallback."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz
from langchain.schema import Document

from config import settings
from domain.value_objects.page_content_type import PageContentType
from infrastructure.ml.ingestion.ocr import ocr_pdf_pages
from infrastructure.ml.ingestion.utils import clean_pdf_text

log = logging.getLogger("detailed")


def pymupdf_table_to_markdown(table) -> str:
    """Convert a PyMuPDF table to markdown format."""
    try:
        data = table.extract()
    except Exception:
        return ""
    if not data or len(data) < 1:
        return ""

    # Normalize column count
    max_cols = max(len(row) for row in data)
    rows = []
    for row in data:
        normalized = [str(cell).strip().replace("|", "\\|") if cell else "" for cell in row]
        while len(normalized) < max_cols:
            normalized.append("")
        rows.append(normalized)

    header = "| " + " | ".join(rows[0]) + " |"
    separator = "|" + "|".join(["---"] * max_cols) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, separator] + body) if body else header


def _extract_page_text(page) -> str:
    """Extract text from page blocks for better multi-column ordering."""
    blocks = page.get_text("blocks")
    blocks.sort(key=lambda b: (round(b[1] / 10) * 10, b[0]))
    return "\n".join(b[4] for b in blocks if len(b) > 4 and b[4].strip())


def _extract_page_tables(page, file_path: Path, page_num: int) -> list[Document]:
    """Detect and extract tables from a page. Returns list of table Documents."""
    docs: list[Document] = []
    try:
        tables = page.find_tables()
        if tables and tables.tables:
            for table in tables.tables:
                md_table = pymupdf_table_to_markdown(table)
                if md_table:
                    docs.append(
                        Document(
                            page_content=md_table,
                            metadata={
                                "page": page_num,
                                "source": str(file_path),
                                "content_type": PageContentType.TABLE.value,
                            },
                        )
                    )
    except Exception:
        pass
    return docs


def _detect_tabular_heuristic(text: str, page_num: int) -> None:
    """Log warning if text has tabular-looking lines but find_tables() found nothing."""
    lines = text.split("\n")
    tabular_lines = sum(
        1 for line in lines if line.count("\t") >= 2 or (len(re.findall(r"  {3,}", line)) > 0)
    )
    if tabular_lines > 3:
        log.warning(
            "Page %d: %d tabular-looking lines but find_tables() found nothing",
            page_num,
            tabular_lines,
        )


def _should_ocr(text: str, min_chars: int, ocr_enabled: bool) -> bool:
    """Return True if OCR should be triggered based on text length and settings."""
    if not text and ocr_enabled:
        return True
    if text and len(text.strip()) < min_chars and ocr_enabled:
        return True
    return False


def _select_best_text(existing_text: str | None, ocr_text: str) -> str:
    """Choose between OCR text and existing short text layer, preferring longer."""
    if existing_text:
        if len(ocr_text) > len(existing_text) * 1.5:
            return ocr_text
        return clean_pdf_text(existing_text) or ocr_text
    return ocr_text


def _process_page_text(text: str, tables_found: bool, page_num: int, file_path: Path) -> Document | None:
    if text and not tables_found:
        _detect_tabular_heuristic(text, page_num)
    if not text or tables_found:
        return None
    cleaned = clean_pdf_text(text)
    if not cleaned:
        return None
    return Document(page_content=cleaned, metadata={"page": page_num, "source": str(file_path)})


def _process_ocr_result(
    page_num: int, ocr_text: str, text_to_compare: dict, file_path: Path,
) -> Document | None:
    if not ocr_text:
        return None
    ocr_text = clean_pdf_text(ocr_text)
    if not ocr_text:
        return None
    final_text = _select_best_text(text_to_compare.get(page_num), ocr_text)
    return Document(page_content=final_text, metadata={"page": page_num, "source": str(file_path)})


def parse_pdf(file_path: Path) -> list[Document]:
    doc = fitz.open(str(file_path))
    pages = []

    ocr_pages_needed = []
    text_to_compare: dict[int, str] = {}

    for page_num in range(1, len(doc) + 1):
        page = doc.load_page(page_num - 1)
        text = _extract_page_text(page)

        table_docs = _extract_page_tables(page, file_path, page_num)
        pages.extend(table_docs)

        min_chars = settings.ocr_min_chars
        if _should_ocr(text, min_chars, settings.ocr_enabled):
            ocr_pages_needed.append(page_num)
            if text:
                text_to_compare[page_num] = text
        else:
            text_doc = _process_page_text(text, bool(table_docs), page_num, file_path)
            if text_doc:
                pages.append(text_doc)

    # --- Batch OCR ---
    if ocr_pages_needed:
        ocr_results = ocr_pdf_pages(doc, ocr_pages_needed, file_path.name)
        for page_num, ocr_text in ocr_results.items():
            ocr_doc = _process_ocr_result(page_num, ocr_text, text_to_compare, file_path)
            if ocr_doc:
                pages.append(ocr_doc)

    doc.close()
    return pages
