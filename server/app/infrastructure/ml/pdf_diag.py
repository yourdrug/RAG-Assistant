"""Assess the actual PDF extraction result, including OCR output."""

from __future__ import annotations

from pathlib import Path

import fitz
from domain.ingestion.text_quality import is_garbled
from domain.ingestion.preview import PDFQualityReport


def assess_pdf_extraction_quality(pdf_path: Path, extracted_docs: list) -> PDFQualityReport:
    """Сравнивает физическое число страниц PDF с тем, что реально извлеклось.

    Страница считается "плохой", если для неё нет ни одного Document в extracted_docs
    (текстовый слой пуст и OCR не справился), либо извлечённый текст мусорный
    (is_garbled). Это именно тот случай ">30% сканов/мусора, и OCR не помог",
    который нужно поймать перед индексацией.
    """
    doc = fitz.open(str(pdf_path))
    try:
        total_pages = len(doc)
    finally:
        doc.close()

    by_page: dict[int, str] = {}

    for d in extracted_docs:
        page_num = d.metadata.get("page")

        if page_num is not None:
            by_page[page_num] = by_page.get(page_num, "") + d.page_content

    n_missing = max(total_pages - len(by_page), 0)
    n_garbled = sum(1 for text in by_page.values() if is_garbled(text))
    # Pages with table content (pipe-separated) should not count as garbled
    n_table = sum(1 for text in by_page.values() if "|" in text and "---" in text)
    n_garbled = max(0, n_garbled - n_table)
    bad = n_missing + n_garbled
    bad_ratio = bad / total_pages if total_pages else 0.0

    return PDFQualityReport(
        total_pages=total_pages,
        n_ok=len(by_page) - n_garbled,
        n_missing=n_missing,
        n_garbled=n_garbled,
        bad_ratio=bad_ratio,
    )
