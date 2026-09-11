"""Characterization tests for ml/ingestion/pdf.py — parse_pdf and helpers.

Creates real PDF files with PyMuPDF to test the full parsing pipeline.
OCR is mocked to avoid PaddleOCR/Surya dependencies in test.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


# ---------------------------------------------------------------------------
# Helpers — create real PDFs
# ---------------------------------------------------------------------------


def _create_pdf(path: Path, pages: list[list[str]], title: str = "") -> Path:
    """Create a PDF with given text per page."""
    doc = fitz.open()
    if title:
        doc.set_metadata({"title": title})
    for page_texts in pages:
        page = doc.new_page()
        y = 72
        for text in page_texts:
            page.insert_text((72, y), text)
            y += 30
    doc.save(str(path))
    doc.close()
    return path


def _create_table_pdf(path: Path) -> Path:
    """Create a PDF with a simple table."""
    doc = fitz.open()
    page = doc.new_page()
    # Insert table-like text
    page.insert_text((72, 72), "Name | Value | Count")
    page.insert_text((72, 102), "Item A | 100 | 5")
    page.insert_text((72, 132), "Item B | 200 | 3")
    doc.save(str(path))
    doc.close()
    return path


def _create_image_heavy_pdf(path: Path) -> Path:
    """Create a PDF with minimal text (simulates scanned page)."""
    doc = fitz.open()
    page = doc.new_page()
    # Very little text — should trigger OCR path
    page.insert_text((72, 72), "X")
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Tests — pymupdf_table_to_markdown
# ---------------------------------------------------------------------------


class TestPymupdfTableToMarkdown:
    def test_simple_table(self):
        from infrastructure.ml.ingestion.pdf import pymupdf_table_to_markdown

        mock_table = MagicMock()
        mock_table.extract.return_value = [["Name", "Value"], ["A", "1"], ["B", "2"]]
        result = pymupdf_table_to_markdown(mock_table)
        assert "| Name |" in result
        assert "| A |" in result
        assert "---" in result

    def test_empty_table(self):
        from infrastructure.ml.ingestion.pdf import pymupdf_table_to_markdown

        mock_table = MagicMock()
        mock_table.extract.return_value = []
        result = pymupdf_table_to_markdown(mock_table)
        assert result == ""

    def test_single_row(self):
        from infrastructure.ml.ingestion.pdf import pymupdf_table_to_markdown

        mock_table = MagicMock()
        mock_table.extract.return_value = [["Col1", "Col2"]]
        result = pymupdf_table_to_markdown(mock_table)
        assert "| Col1 |" in result
        # No body rows, just header
        lines = result.strip().split("\n")
        assert len(lines) == 1  # only header

    def test_exception_returns_empty(self):
        from infrastructure.ml.ingestion.pdf import pymupdf_table_to_markdown

        mock_table = MagicMock()
        mock_table.extract.side_effect = Exception("fail")
        result = pymupdf_table_to_markdown(mock_table)
        assert result == ""


# ---------------------------------------------------------------------------
# Tests — _order_blocks_columnwise
# ---------------------------------------------------------------------------


class TestOrderBlocksColumnwise:
    def test_single_column(self):
        from infrastructure.ml.ingestion.pdf import _order_blocks_columnwise

        blocks = [(10, 10, 100, 30, "Line 1"), (10, 40, 100, 60, "Line 2")]
        result = _order_blocks_columnwise(blocks, 500)
        assert result == ["Line 1", "Line 2"]

    def test_two_columns(self):
        from infrastructure.ml.ingestion.pdf import _order_blocks_columnwise

        # Column 1 at x=10, Column 2 at x=300 (gap > 8% of 500 = 40)
        blocks = [
            (10, 10, 100, 30, "Col1-Line1"),
            (300, 10, 400, 30, "Col2-Line1"),
            (10, 40, 100, 60, "Col1-Line2"),
            (300, 40, 400, 60, "Col2-Line2"),
        ]
        result = _order_blocks_columnwise(blocks, 500)
        assert result == ["Col1-Line1", "Col1-Line2", "Col2-Line1", "Col2-Line2"]

    def test_empty_blocks(self):
        from infrastructure.ml.ingestion.pdf import _order_blocks_columnwise

        assert _order_blocks_columnwise([], 500) == []

    def test_single_block(self):
        from infrastructure.ml.ingestion.pdf import _order_blocks_columnwise

        blocks = [(10, 10, 100, 30, "only block")]
        result = _order_blocks_columnwise(blocks, 500)
        assert result == ["only block"]


# ---------------------------------------------------------------------------
# Tests — _should_ocr
# ---------------------------------------------------------------------------


class TestShouldOcr:
    def test_empty_text_with_ocr_enabled(self):
        from infrastructure.ml.ingestion.pdf import _should_ocr

        assert _should_ocr("", 100, True) is True

    def test_short_text_with_ocr_enabled(self):
        from infrastructure.ml.ingestion.pdf import _should_ocr

        assert _should_ocr("hi", 100, True) is True

    def test_long_text_not_ocr(self):
        from infrastructure.ml.ingestion.pdf import _should_ocr

        assert _should_ocr("x" * 200, 100, True) is False

    def test_ocr_disabled(self):
        from infrastructure.ml.ingestion.pdf import _should_ocr

        assert _should_ocr("", 100, False) is False
        assert _should_ocr("hi", 100, False) is False


# ---------------------------------------------------------------------------
# Tests — _select_best_text
# ---------------------------------------------------------------------------


class TestSelectBestText:
    def test_no_existing(self):
        from infrastructure.ml.ingestion.pdf import _select_best_text

        assert _select_best_text(None, "ocr text") == "ocr text"

    def test_ocr_much_longer(self):
        from infrastructure.ml.ingestion.pdf import _select_best_text

        result = _select_best_text("short", "a" * 100)
        assert result == "a" * 100

    def test_existing_comparable(self):
        from infrastructure.ml.ingestion.pdf import _select_best_text

        result = _select_best_text("existing text here", "ocr")
        # existing is not 1.5x shorter than ocr, so returns cleaned existing
        assert "existing" in result or result == "ocr"


# ---------------------------------------------------------------------------
# Tests — _compute_text_quality
# ---------------------------------------------------------------------------


class TestComputeTextQuality:
    def test_page_with_text(self):
        from infrastructure.ml.ingestion.pdf import _compute_text_quality

        mock_page = MagicMock()
        mock_page.rect.width = 500
        mock_page.rect.height = 700
        mock_page.get_text.return_value = "word " * 50  # 250 chars
        result = _compute_text_quality(mock_page)
        assert 0.0 <= result <= 1.0

    def test_empty_page(self):
        from infrastructure.ml.ingestion.pdf import _compute_text_quality

        mock_page = MagicMock()
        mock_page.rect.width = 500
        mock_page.rect.height = 700
        mock_page.get_text.return_value = ""
        result = _compute_text_quality(mock_page)
        assert result == 0.0

    def test_exception_returns_zero(self):
        from infrastructure.ml.ingestion.pdf import _compute_text_quality

        mock_page = MagicMock()
        mock_page.rect.width = -1
        mock_page.rect.height = -1
        result = _compute_text_quality(mock_page)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Tests — _extract_doc_metadata
# ---------------------------------------------------------------------------


class TestExtractDocMetadata:
    def test_with_metadata(self):
        from infrastructure.ml.ingestion.pdf import _extract_doc_metadata

        mock_doc = MagicMock()
        mock_doc.metadata = {"title": "My Title", "author": "Author", "subject": "", "keywords": ""}
        mock_doc.__len__ = MagicMock(return_value=5)
        result = _extract_doc_metadata(mock_doc)
        assert result["doc_title"] == "My Title"
        assert result["doc_author"] == "Author"
        assert result["page_count"] == 5
        assert "doc_subject" not in result

    def test_empty_metadata(self):
        from infrastructure.ml.ingestion.pdf import _extract_doc_metadata

        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_doc.__len__ = MagicMock(return_value=3)
        result = _extract_doc_metadata(mock_doc)
        assert "doc_title" not in result
        assert result["page_count"] == 3


# ---------------------------------------------------------------------------
# Tests — _find_boilerplate_patterns
# ---------------------------------------------------------------------------


class TestFindBoilerplatePatterns:
    def test_few_pages_returns_empty(self):
        from infrastructure.ml.ingestion.pdf import _find_boilerplate_patterns

        # Less than _BOILERPLATE_MIN_PAGES (4)
        classified = [
            ([[(0, 0, 100, 20, "Header")], [], []]),
        ]
        result = _find_boilerplate_patterns(classified)
        assert result == set()

    def test_recurring_pattern_detected(self):
        from infrastructure.ml.ingestion.pdf import _find_boilerplate_patterns

        # 5 pages all with same header
        header_block = [(0, 0, 100, 20, "Running Header")]
        classified = [
            (header_block, [[(0, 30, 100, 100, "Body text")]], []),
            (header_block, [[(0, 30, 100, 100, "Body text")]], []),
            (header_block, [[(0, 30, 100, 100, "Body text")]], []),
            (header_block, [[(0, 30, 100, 100, "Body text")]], []),
            (header_block, [[(0, 30, 100, 100, "Body text")]], []),
        ]
        result = _find_boilerplate_patterns(classified)
        assert "Running Header" in result


# ---------------------------------------------------------------------------
# Tests — parse_pdf (real PDF, mocked OCR)
# ---------------------------------------------------------------------------


class TestParsePdf:
    def test_simple_text_pdf(self, tmp_path):
        from infrastructure.ml.ingestion.pdf import parse_pdf

        pdf_path = _create_pdf(
            tmp_path / "test.pdf",
            [["Hello world", "This is a test document with enough text to pass the minimum threshold."]],
        )
        with patch("infrastructure.ml.ingestion.pdf.settings") as mock_settings:
            mock_settings.ocr_enabled = False
            mock_settings.ocr_min_chars = 50
            result = parse_pdf(pdf_path)

        assert len(result) >= 1
        assert any("Hello world" in d.page_content for d in result)

    def test_multi_page_pdf(self, tmp_path):
        from infrastructure.ml.ingestion.pdf import parse_pdf

        pdf_path = _create_pdf(
            tmp_path / "multi.pdf",
            [
                ["Page one content with enough text to be detected."],
                ["Page two content with enough text to be detected."],
            ],
        )
        with patch("infrastructure.ml.ingestion.pdf.settings") as mock_settings:
            mock_settings.ocr_enabled = False
            mock_settings.ocr_min_chars = 50
            result = parse_pdf(pdf_path)

        assert len(result) >= 2
        pages = {d.metadata.get("page") for d in result}
        assert 1 in pages
        assert 2 in pages

    def test_metadata_has_source(self, tmp_path):
        from infrastructure.ml.ingestion.pdf import parse_pdf

        pdf_path = _create_pdf(tmp_path / "meta.pdf", [["Some text content here for testing."]])
        with patch("infrastructure.ml.ingestion.pdf.settings") as mock_settings:
            mock_settings.ocr_enabled = False
            mock_settings.ocr_min_chars = 10
            result = parse_pdf(pdf_path)

        assert len(result) >= 1
        assert result[0].metadata["source"] == str(pdf_path)

    def test_metadata_has_text_quality(self, tmp_path):
        from infrastructure.ml.ingestion.pdf import parse_pdf

        pdf_path = _create_pdf(tmp_path / "tq.pdf", [["Text content for quality check."]])
        with patch("infrastructure.ml.ingestion.pdf.settings") as mock_settings:
            mock_settings.ocr_enabled = False
            mock_settings.ocr_min_chars = 10
            result = parse_pdf(pdf_path)

        assert "text_quality" in result[0].metadata

    def test_metadata_has_image_info(self, tmp_path):
        from infrastructure.ml.ingestion.pdf import parse_pdf

        pdf_path = _create_pdf(tmp_path / "img.pdf", [["Image info test."]])
        with patch("infrastructure.ml.ingestion.pdf.settings") as mock_settings:
            mock_settings.ocr_enabled = False
            mock_settings.ocr_min_chars = 10
            result = parse_pdf(pdf_path)

        assert "has_images" in result[0].metadata
        assert "image_count" in result[0].metadata

    def test_pdf_title_in_metadata(self, tmp_path):
        from infrastructure.ml.ingestion.pdf import parse_pdf

        pdf_path = _create_pdf(tmp_path / "titled.pdf", [["Title test."]], title="My Document")
        with patch("infrastructure.ml.ingestion.pdf.settings") as mock_settings:
            mock_settings.ocr_enabled = False
            mock_settings.ocr_min_chars = 10
            result = parse_pdf(pdf_path)

        assert result[0].metadata.get("doc_title") == "My Document"

    def test_empty_text_pdf_returns_empty(self, tmp_path):
        """PDF with no extractable text should return empty list (no OCR)."""
        from infrastructure.ml.ingestion.pdf import parse_pdf

        # Create a PDF with truly empty pages
        doc = fitz.open()
        doc.new_page()
        doc.save(str(tmp_path / "empty.pdf"))
        doc.close()

        with patch("infrastructure.ml.ingestion.pdf.settings") as mock_settings:
            mock_settings.ocr_enabled = False
            mock_settings.ocr_min_chars = 50
            result = parse_pdf(tmp_path / "empty.pdf")

        assert len(result) == 0
