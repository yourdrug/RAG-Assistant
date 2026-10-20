"""Regression coverage for OCR page results returned to preview callers."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from application.documents.diagnostics import PDFDiagnosticService
from domain.ingestion.preview import DryRunPageResult
from domain.knowledge.enums import PageContentType


@pytest.mark.parametrize(
    "ocr_text,cleaned_text,expected_type",
    [
        ("  Заявление  ", "Заявление", PageContentType.TEXT),
        ("", "", PageContentType.SCAN),
        ("noise", "", PageContentType.SCAN),
    ],
)
def test_ocr_returns_updated_pages_and_preserves_other_pages(
    ocr_text: str, cleaned_text: str, expected_type: PageContentType
) -> None:
    cleaner = Mock()
    cleaner.clean.return_value = cleaned_text
    ocr = Mock()
    ocr.ocr_pages.return_value = {2: ocr_text}
    pdf = Mock()
    service = PDFDiagnosticService(
        classifier=Mock(),
        text_cleaner=cleaner,
        ocr=ocr,
        pdf_doc=pdf,
        storage=Mock(),
        max_dry_run_bytes=1024,
    )
    unchanged = DryRunPageResult(page=1, type=PageContentType.TEXT, chars=5, full_text="Текст")
    original = DryRunPageResult(page=2, type=PageContentType.EMPTY)
    pages = [unchanged, original]
    path = Path("scan.pdf")

    result, counts, total_chars = service.ocr_problem_pages(path, pages)

    assert result[0] is unchanged
    assert result[1].page == 2
    assert result[1].type == expected_type
    assert result[1].content_type == PageContentType.OCR
    assert result[1].previous_type == PageContentType.EMPTY
    assert result[1].full_text == cleaned_text
    assert result[1].preview == cleaned_text[:200]
    assert result[1].chars == len(cleaned_text)
    assert total_chars == sum(page.chars for page in result)
    assert sum(counts.values()) == len(result)

    for page_type in PageContentType:
        assert counts[page_type] == sum(page.type == page_type for page in result)

    assert pages == [unchanged, original]
    ocr.ocr_pages.assert_called_once_with(pdf.open.return_value, [2], path.name)
    pdf.close.assert_called_once_with(pdf.open.return_value)
