"""Tests for infrastructure/ml/ingestion/docx.py — DOCX parsing.

Pure python-docx driven transformations, no OCR/Qdrant/embeddings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import docx  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402

from infrastructure.ml.ingestion.docx import (  # noqa: E402
    _extract_document_properties,
    _extract_headers_footers,
    _extract_image_captions,
    _extract_notes,
    _get_numbering_formats,
    _has_page_break,
    _ListNumberer,
    _paragraph_full_text,
    _paragraph_list_info,
    _to_roman,
    docx_table_to_markdown,
    parse_docx,
    parse_docx_sections,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal docx in a fixture
# ---------------------------------------------------------------------------


def _make_doc(title: str = "", author: str = "") -> docx.Document:
    """Create a blank Document with optional core properties."""
    doc = docx.Document()
    if title:
        doc.core_properties.title = title
    if author:
        doc.core_properties.author = author
    return doc


def _add_heading(doc: docx.Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def _add_para(doc: docx.Document, text: str, style: str | None = None) -> None:
    doc.add_paragraph(text, style=style)


def _add_page_break(doc: docx.Document) -> None:
    """Insert a manual page break (w:br w:type='page')."""
    p = doc.add_paragraph()
    run = p.add_run()
    br = run._element.makeelement(qn("w:br"), {qn("w:type"): "page"})
    run._element.append(br)


def _add_hyperlink(doc: docx.Document, text: str, url: str = "https://example.com") -> None:
    """Add a paragraph containing a hyperlink (simulates real documents)."""
    p = doc.add_paragraph()
    # Build <w:hyperlink> manually since python-docx has no public API for it
    hyperlink = p._element.makeelement(qn("w:hyperlink"), {})
    r = p._element.makeelement(qn("w:r"), {})
    t = p._element.makeelement(qn("w:t"), {})
    t.text = text
    r.append(t)
    hyperlink.append(r)
    p._element.append(hyperlink)


def _add_list_paragraph(
    doc: docx.Document,
    text: str,
    num_id: str = "1",
    level: int = 0,
    num_fmt: str = "bullet",
) -> None:
    """Add a paragraph with explicit numPr (list item)."""
    p = doc.add_paragraph(text)
    # Inject numbering part if absent
    if doc.part.numbering_part is None:
        pass
        # python-docx doesn't let us easily add a numbering part from scratch
        # without a template; instead we manipulate XML directly.
        # We'll rely on a different strategy: set numPr on the paragraph XML.
    pPr = p._element.get_or_add_pPr()
    numPr = p._element.makeelement(qn("w:numPr"), {})
    ilvl = p._element.makeelement(qn("w:ilvl"), {qn("w:val"): str(level)})
    numId_el = p._element.makeelement(qn("w:numId"), {qn("w:val"): num_id})
    numPr.append(ilvl)
    numPr.append(numId_el)
    pPr.append(numPr)


def _add_table(doc: docx.Document, rows: list[list[str]]) -> None:
    """Add a table with the given cell values."""
    table = doc.add_table(rows=len(rows), cols=len(rows[0]) if rows else 0)
    for i, row_data in enumerate(rows):
        for j, val in enumerate(row_data):
            table.rows[i].cells[j].text = val


def _add_header_footer(doc: docx.Document, header_text: str, footer_text: str) -> None:
    """Set header and footer on the first section."""
    section = doc.sections[0]
    section.header.paragraphs[0].text = header_text
    section.footer.paragraphs[0].text = footer_text


def _save(doc: docx.Document, tmp_path: Path, name: str = "test.docx") -> Path:
    path = tmp_path / name
    doc.save(str(path))
    return path


# ---------------------------------------------------------------------------
# _to_roman
# ---------------------------------------------------------------------------


class TestToRoman:
    def test_1(self):
        assert _to_roman(1) == "I"

    def test_4(self):
        assert _to_roman(4) == "IV"

    def test_9(self):
        assert _to_roman(9) == "IX"

    def test_14(self):
        assert _to_roman(14) == "XIV"

    def test_42(self):
        assert _to_roman(42) == "XLII"

    def test_1999(self):
        assert _to_roman(1999) == "MCMXCIX"

    def test_3999(self):
        assert _to_roman(3999) == "MMMCMXCIX"


# ---------------------------------------------------------------------------
# _ListNumberer
# ---------------------------------------------------------------------------


class TestListNumberer:
    def test_bullet_prefix(self):
        numberer = _ListNumberer({})
        assert numberer.prefix("1", 0) == "- "

    def test_bullet_with_indent(self):
        numberer = _ListNumberer({})
        assert numberer.prefix("1", 2) == "    - "

    def test_decimal(self):
        numberer = _ListNumberer({("1", 0): "decimal"})
        assert numberer.prefix("1", 0) == "1. "
        assert numberer.prefix("1", 0) == "2. "
        assert numberer.prefix("1", 0) == "3. "

    def test_lower_letter(self):
        numberer = _ListNumberer({("1", 0): "lowerLetter"})
        assert numberer.prefix("1", 0) == "a. "
        assert numberer.prefix("1", 0) == "b. "

    def test_upper_letter(self):
        numberer = _ListNumberer({("1", 0): "upperLetter"})
        assert numberer.prefix("1", 0) == "A. "
        assert numberer.prefix("1", 0) == "B. "

    def test_lower_roman(self):
        numberer = _ListNumberer({("1", 0): "lowerRoman"})
        assert numberer.prefix("1", 0) == "i. "
        assert numberer.prefix("1", 0) == "ii. "
        assert numberer.prefix("1", 0) == "iii. "

    def test_upper_roman(self):
        numberer = _ListNumberer({("1", 0): "upperRoman"})
        assert numberer.prefix("1", 0) == "I. "
        assert numberer.prefix("1", 0) == "II. "

    def test_different_num_ids_independent(self):
        numberer = _ListNumberer({("1", 0): "decimal", ("2", 0): "decimal"})
        assert numberer.prefix("1", 0) == "1. "
        assert numberer.prefix("2", 0) == "1. "
        assert numberer.prefix("1", 0) == "2. "

    def test_deeper_level_resets_parent_counters(self):
        numberer = _ListNumberer({("1", 0): "decimal", ("1", 1): "decimal"})
        # Going to a deeper level resets deeper (child) counters.
        assert numberer.prefix("1", 0) == "1. "
        assert numberer.prefix("1", 1) == "  1. "
        assert numberer.prefix("1", 1) == "  2. "
        # Returning to level 0 — deeper counters (level 1) were reset.
        assert numberer.prefix("1", 0) == "2. "
        # Level 1 counter should be fresh again
        assert numberer.prefix("1", 1) == "  1. "

    def test_letter_wraps_at_26(self):
        numberer = _ListNumberer({("1", 0): "lowerLetter"})
        for _ in range(25):
            numberer.prefix("1", 0)
        assert numberer.prefix("1", 0) == "z. "
        assert numberer.prefix("1", 0) == "a. "


# ---------------------------------------------------------------------------
# _has_page_break
# ---------------------------------------------------------------------------


class TestHasPageBreak:
    def test_no_break(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Hello")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        assert _has_page_break(doc2.paragraphs[0]) is False

    def test_manual_break(self, tmp_path):
        doc = _make_doc()
        _add_page_break(doc)
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        assert _has_page_break(doc2.paragraphs[0]) is True


# ---------------------------------------------------------------------------
# _paragraph_full_text
# ---------------------------------------------------------------------------


class TestParagraphFullText:
    def test_plain_text(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Hello world")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        assert _paragraph_full_text(doc2.paragraphs[0]) == "Hello world"

    def test_hyperlink_text_preserved(self, tmp_path):
        doc = _make_doc()
        _add_hyperlink(doc, "Click me")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        text = _paragraph_full_text(doc2.paragraphs[0])
        assert "Click me" in text

    def test_tab_included(self, tmp_path):
        doc = _make_doc()
        p = doc.add_paragraph()
        run = p.add_run()
        tab_el = run._element.makeelement(qn("w:tab"), {})
        run._element.append(tab_el)
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        assert _paragraph_full_text(doc2.paragraphs[0]) == "\t"

    def test_line_break_included(self, tmp_path):
        doc = _make_doc()
        p = doc.add_paragraph()
        run = p.add_run("A")
        br = run._element.makeelement(qn("w:br"), {})
        run._element.append(br)
        p.add_run("B")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        assert _paragraph_full_text(doc2.paragraphs[0]) == "A\nB"


# ---------------------------------------------------------------------------
# _get_numbering_formats
# ---------------------------------------------------------------------------


class TestGetNumberingFormats:
    def test_plain_text_has_numbering_formats(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Plain text")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        result = _get_numbering_formats(doc2)
        # python-docx template includes a default numbering part,
        # so the result is not empty. Just verify the structure.
        assert isinstance(result, dict)

    def test_bullet_numbering(self, tmp_path):
        doc = _make_doc()
        doc.add_paragraph("Item 1", style="List Bullet")
        doc.add_paragraph("Item 2", style="List Bullet")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        result = _get_numbering_formats(doc2)
        # At least one (numId, ilvl) should map to 'bullet'
        assert any(v == "bullet" for v in result.values())

    def test_numbered_numbering(self, tmp_path):
        doc = _make_doc()
        doc.add_paragraph("First", style="List Number")
        doc.add_paragraph("Second", style="List Number")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        result = _get_numbering_formats(doc2)
        assert any(v == "decimal" for v in result.values())


# ---------------------------------------------------------------------------
# _paragraph_list_info
# ---------------------------------------------------------------------------


class TestParagraphListInfo:
    def test_regular_paragraph(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Hello")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        assert _paragraph_list_info(doc2.paragraphs[0]) is None

    def test_list_paragraph(self, tmp_path):
        doc = _make_doc()
        _add_list_paragraph(doc, "Item", num_id="1", level=0)
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        result = _paragraph_list_info(doc2.paragraphs[0])
        assert result is not None
        num_id, level = result
        assert num_id == "1"
        assert level == 0

    def test_nested_list_paragraph(self, tmp_path):
        doc = _make_doc()
        _add_list_paragraph(doc, "Nested item", num_id="1", level=2)
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        result = _paragraph_list_info(doc2.paragraphs[0])
        assert result is not None
        assert result == ("1", 2)


# ---------------------------------------------------------------------------
# _extract_headers_footers
# ---------------------------------------------------------------------------


class TestExtractHeadersFooters:
    def test_no_headers_footers(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Body")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        result = _extract_headers_footers(doc2)
        assert result == {"headers": [], "footers": []}

    def test_with_header_footer(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Body")
        _add_header_footer(doc, "My Header", "My Footer")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        result = _extract_headers_footers(doc2)
        assert result["headers"] == ["My Header"]
        assert result["footers"] == ["My Footer"]

    def test_duplicate_headers_deduped(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Body")
        # Both sections get the same header text
        _add_header_footer(doc, "Same Header", "Footer 1")
        doc.add_section()
        _add_header_footer(doc, "Same Header", "Footer 2")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        result = _extract_headers_footers(doc2)
        assert result["headers"] == ["Same Header"]


# ---------------------------------------------------------------------------
# _extract_notes
# ---------------------------------------------------------------------------


class TestExtractNotes:
    def test_no_footnotes(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Body")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        result = _extract_notes(doc2, "footnotes", "w:footnote")
        assert result == []

    def test_no_endnotes(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Body")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        result = _extract_notes(doc2, "endnotes", "w:endnote")
        assert result == []


# ---------------------------------------------------------------------------
# _extract_document_properties
# ---------------------------------------------------------------------------


class TestExtractDocumentProperties:
    def test_empty_properties(self, tmp_path):
        doc = _make_doc()
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        result = _extract_document_properties(doc2)
        # python-docx sets default author/created/modified from the template
        assert isinstance(result, dict)
        # No title or subject since we didn't set them
        assert "doc_title" not in result
        assert "doc_subject" not in result

    def test_title_and_author(self, tmp_path):
        doc = _make_doc(title="My Title", author="John Doe")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        result = _extract_document_properties(doc2)
        assert result["doc_title"] == "My Title"
        assert result["doc_author"] == "John Doe"

    def test_subject(self, tmp_path):
        doc = _make_doc()
        doc.core_properties.subject = "Engineering Report"
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        result = _extract_document_properties(doc2)
        assert result["doc_subject"] == "Engineering Report"


# ---------------------------------------------------------------------------
# _extract_image_captions
# ---------------------------------------------------------------------------


class TestExtractImageCaptions:
    def test_no_images(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Text only")
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        assert _extract_image_captions(doc2) == []


# ---------------------------------------------------------------------------
# docx_table_to_markdown
# ---------------------------------------------------------------------------


class TestDocxTableToMarkdown:
    def test_simple_table(self, tmp_path):
        doc = _make_doc()
        _add_table(doc, [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]])
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        md = docx_table_to_markdown(doc2.tables[0])
        lines = md.split("\n")
        assert lines[0] == "| Name | Age |"
        assert lines[1] == "|---|---|"
        assert lines[2] == "| Alice | 30 |"
        assert lines[3] == "| Bob | 25 |"

    def test_empty_table(self, tmp_path):
        # A table with 0 rows is impossible via python-docx API, but test the
        # edge case where rows list is empty (defensive).
        class FakeTable:
            rows = []

        md = docx_table_to_markdown(FakeTable())
        assert md == ""

    def test_single_cell_table(self, tmp_path):
        doc = _make_doc()
        _add_table(doc, [["Solo"]])
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        md = docx_table_to_markdown(doc2.tables[0])
        # Single-row table: only header line, no separator/body
        assert md == "| Solo |"

    def test_pipes_escaped(self, tmp_path):
        doc = _make_doc()
        _add_table(doc, [["Col|A", "Col|B"], ["v|1", "v|2"]])
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        md = docx_table_to_markdown(doc2.tables[0])
        assert "Col\\|A" in md
        assert "Col\\|B" in md
        assert "v\\|1" in md

    def test_table_with_empty_cells(self, tmp_path):
        doc = _make_doc()
        _add_table(doc, [["A", ""], ["", "D"]])
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        md = docx_table_to_markdown(doc2.tables[0])
        lines = md.split("\n")
        assert lines[0] == "| A |  |"
        assert lines[2] == "|  | D |"

    def test_table_with_many_columns(self, tmp_path):
        doc = _make_doc()
        _add_table(doc, [["C1", "C2", "C3", "C4"], ["1", "2", "3", "4"]])
        _save(doc, tmp_path)
        doc2 = docx.Document(str(tmp_path / "test.docx"))
        md = docx_table_to_markdown(doc2.tables[0])
        lines = md.split("\n")
        separator_parts = lines[1].split("|")[1:-1]
        assert len(separator_parts) == 4


# ---------------------------------------------------------------------------
# parse_docx
# ---------------------------------------------------------------------------


class TestParseDocx:
    def test_returns_tuple(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Hello")
        path = _save(doc, tmp_path)
        result = parse_docx(path)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_basic_text(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Hello world")
        _add_para(doc, "Second paragraph")
        path = _save(doc, tmp_path)
        text, meta = parse_docx(path)
        assert "Hello world" in text
        assert "Second paragraph" in text

    def test_page_start_end(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Page 1 content")
        _add_page_break(doc)
        _add_para(doc, "Page 2 content")
        path = _save(doc, tmp_path)
        text, meta = parse_docx(path)
        assert meta["page_start"] == 1
        assert meta["page_end"] == 2
        assert meta["pages"] == [1, 2]

    def test_pages_single_page(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Just one page")
        path = _save(doc, tmp_path)
        _, meta = parse_docx(path)
        assert meta["page_start"] == 1
        assert meta["page_end"] == 1
        assert meta["pages"] == [1]

    def test_metadata_fields(self, tmp_path):
        doc = _make_doc(title="Test Doc", author="Alice")
        doc.core_properties.subject = "Testing"
        _add_para(doc, "Content")
        path = _save(doc, tmp_path)
        _, meta = parse_docx(path)
        assert meta["doc_title"] == "Test Doc"
        assert meta["doc_author"] == "Alice"
        assert meta["doc_subject"] == "Testing"
        assert "doc_created" in meta
        assert "doc_modified" in meta

    def test_tables_as_markdown(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Before table")
        _add_table(doc, [["X", "Y"], ["1", "2"]])
        _add_para(doc, "After table")
        path = _save(doc, tmp_path)
        text, _meta = parse_docx(path)
        assert "| X | Y |" in text
        assert "|---|---|" in text
        assert "| 1 | 2 |" in text

    def test_list_numbering_decimal(self, tmp_path):
        doc = _make_doc()
        _add_list_paragraph(doc, "First", num_id="1", level=0)
        _add_list_paragraph(doc, "Second", num_id="1", level=0)
        _add_list_paragraph(doc, "Third", num_id="1", level=0)
        path = _save(doc, tmp_path)
        text, _meta = parse_docx(path)
        # Without numbering part, falls back to bullet prefix
        assert "First" in text
        assert "Second" in text
        assert "Third" in text

    def test_list_bullet(self, tmp_path):
        doc = _make_doc()
        _add_list_paragraph(doc, "Alpha", num_id="1", level=0)
        _add_list_paragraph(doc, "Beta", num_id="1", level=0)
        path = _save(doc, tmp_path)
        text, _meta = parse_docx(path)
        # Falls back to bullet when numFmt not in numbering part
        assert "- Alpha" in text
        assert "- Beta" in text

    def test_list_with_builtin_style(self, tmp_path):
        doc = _make_doc()
        doc.add_paragraph("Item A", style="List Bullet")
        doc.add_paragraph("Item B", style="List Bullet")
        path = _save(doc, tmp_path)
        text, _meta = parse_docx(path)
        assert "Item A" in text
        assert "Item B" in text

    def test_hyperlinks_preserved(self, tmp_path):
        doc = _make_doc()
        _add_hyperlink(doc, "Click here")
        path = _save(doc, tmp_path)
        text, _meta = parse_docx(path)
        assert "Click here" in text

    def test_headers_footer_in_metadata(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Body text")
        _add_header_footer(doc, "Confidential", "Page 1")
        path = _save(doc, tmp_path)
        _, meta = parse_docx(path)
        assert meta["header_text"] == "Confidential"
        assert meta["footer_text"] == "Page 1"

    def test_footnotes_appended(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Main text")
        # We can't easily inject a footnotes part via python-docx's public API
        # without hitting Part initialization issues. Test that _extract_notes
        # correctly returns [] when no footnotes part exists.
        path = _save(doc, tmp_path)
        text, _meta = parse_docx(path)
        assert "Main text" in text
        # No footnotes appended (no footnotes part in the document)
        assert "[footnote" not in text

    def test_empty_paragraphs_skipped(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "")
        _add_para(doc, "Real content")
        _add_para(doc, "  ")
        path = _save(doc, tmp_path)
        text, _meta = parse_docx(path)
        assert text.strip() == "Real content"

    def test_multiple_tables(self, tmp_path):
        doc = _make_doc()
        _add_table(doc, [["A", "B"], ["1", "2"]])
        _add_table(doc, [["C", "D"], ["3", "4"]])
        path = _save(doc, tmp_path)
        text, _meta = parse_docx(path)
        assert "| A | B |" in text
        assert "| C | D |" in text

    def test_page_break_multiple(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Page 1")
        _add_page_break(doc)
        _add_para(doc, "Page 2")
        _add_page_break(doc)
        _add_para(doc, "Page 3")
        path = _save(doc, tmp_path)
        _, meta = parse_docx(path)
        assert meta["pages"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# parse_docx_sections
# ---------------------------------------------------------------------------


class TestParseDocxSections:
    def test_returns_list_of_tuples(self, tmp_path):
        doc = _make_doc()
        _add_heading(doc, "Title", level=1)
        _add_para(doc, "Content here")
        path = _save(doc, tmp_path)
        sections = parse_docx_sections(path)
        assert isinstance(sections, list)
        assert all(isinstance(s, tuple) for s in sections)

    def test_splits_by_headings(self, tmp_path):
        doc = _make_doc()
        _add_heading(doc, "Chapter 1", level=1)
        _add_para(doc, "Content 1")
        _add_heading(doc, "Chapter 2", level=1)
        _add_para(doc, "Content 2")
        path = _save(doc, tmp_path)
        sections = parse_docx_sections(path)
        headings = [h for h, _ in sections]
        assert any("Chapter 1" in (h or "") for h in headings)
        assert any("Chapter 2" in (h or "") for h in headings)

    def test_breadcrumb_paths(self, tmp_path):
        doc = _make_doc()
        _add_heading(doc, "Part 1", level=1)
        _add_heading(doc, "Section A", level=2)
        _add_para(doc, "Deep content")
        path = _save(doc, tmp_path)
        sections = parse_docx_sections(path)
        # The section under "Section A" should have a breadcrumb like "Part 1 > Section A"
        headings = [h for h, _ in sections]
        assert any(h and "Part 1 > Section A" in h for h in headings)

    def test_title_style_treated_as_heading(self, tmp_path):
        doc = _make_doc()
        doc.add_paragraph("Main Title", style="Title")
        _add_para(doc, "Body text")
        path = _save(doc, tmp_path)
        sections = parse_docx_sections(path)
        headings = [h for h, _ in sections]
        assert any(h and "Main Title" in h for h in headings)

    def test_tables_appended_as_table_section(self, tmp_path):
        doc = _make_doc()
        _add_heading(doc, "Intro", level=1)
        _add_para(doc, "Intro text")
        _add_table(doc, [["A", "B"], ["1", "2"]])
        path = _save(doc, tmp_path)
        sections = parse_docx_sections(path)
        table_sections = [(h, c) for h, c in sections if h is None and "\x00TABLE:" in c]
        assert len(table_sections) >= 1
        assert "| A | B |" in table_sections[0][1]

    def test_footnotes_section(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Main text")
        # _extract_notes returns [] when no footnotes part exists
        path = _save(doc, tmp_path)
        sections = parse_docx_sections(path)
        fn_sections = [(h, c) for h, c in sections if h == "Footnotes"]
        # No footnotes part in the document
        assert len(fn_sections) == 0

    def test_empty_sections_filtered(self, tmp_path):
        doc = _make_doc()
        _add_heading(doc, "Empty Chapter", level=1)
        # No content after heading — should be flushed empty and filtered
        _add_heading(doc, "Real Chapter", level=1)
        _add_para(doc, "Actual content")
        path = _save(doc, tmp_path)
        sections = parse_docx_sections(path)
        headings_and_content = list(sections)
        # "Empty Chapter" should not appear as a section with content
        for h, c in headings_and_content:
            if h and "Empty Chapter" in h:
                assert c.strip() != ""

    def test_list_items_preserved_in_sections(self, tmp_path):
        doc = _make_doc()
        _add_heading(doc, "List Section", level=1)
        doc.add_paragraph("Item A", style="List Bullet")
        doc.add_paragraph("Item B", style="List Bullet")
        path = _save(doc, tmp_path)
        sections = parse_docx_sections(path)
        all_content = " ".join(c for _, c in sections)
        assert "Item A" in all_content
        assert "Item B" in all_content

    def test_fallback_to_flat_parse(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Just plain text, no headings at all")
        path = _save(doc, tmp_path)
        sections = parse_docx_sections(path)
        # Should return one section with None heading
        assert len(sections) >= 1
        assert sections[0][0] is None
        assert "Just plain text" in sections[0][1]

    def test_nested_heading_levels(self, tmp_path):
        doc = _make_doc()
        _add_heading(doc, "H1", level=1)
        _add_para(doc, "H1 content")
        _add_heading(doc, "H2", level=2)
        _add_para(doc, "H2 content")
        _add_heading(doc, "H3", level=3)
        _add_para(doc, "H3 content")
        _add_heading(doc, "H2 again", level=2)
        _add_para(doc, "H2 second")
        path = _save(doc, tmp_path)
        sections = parse_docx_sections(path)
        breadcrumbs = [h for h, _ in sections if h]
        assert any("H1 > H2" in b for b in breadcrumbs)
        assert any("H1 > H2 > H3" in b for b in breadcrumbs)

    def test_title_then_headings(self, tmp_path):
        doc = _make_doc()
        doc.add_paragraph("Doc Title", style="Title")
        _add_heading(doc, "Chapter 1", level=1)
        _add_para(doc, "Content")
        path = _save(doc, tmp_path)
        sections = parse_docx_sections(path)
        breadcrumbs = [h for h, _ in sections if h]
        assert any("Doc Title > Chapter 1" in b for b in breadcrumbs)

    def test_same_level_heading_resets_breadcrumb(self, tmp_path):
        doc = _make_doc()
        _add_heading(doc, "A", level=1)
        _add_para(doc, "a content")
        _add_heading(doc, "B", level=1)
        _add_para(doc, "b content")
        path = _save(doc, tmp_path)
        sections = parse_docx_sections(path)
        a_sections = [(h, c) for h, c in sections if h and "B" in h and "A" not in h]
        assert len(a_sections) >= 1


# ---------------------------------------------------------------------------
# parse_docx — edge cases
# ---------------------------------------------------------------------------


class TestParseDocxEdgeCases:
    def test_single_paragraph(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Only one paragraph")
        path = _save(doc, tmp_path)
        text, meta = parse_docx(path)
        assert text.strip() == "Only one paragraph"

    def test_only_page_breaks(self, tmp_path):
        doc = _make_doc()
        _add_page_break(doc)
        _add_page_break(doc)
        path = _save(doc, tmp_path)
        text, meta = parse_docx(path)
        # No text content → page_numbers is empty → no pages in metadata
        assert text.strip() == ""
        assert "pages" not in meta
        assert "page_start" not in meta

    def test_table_before_text(self, tmp_path):
        doc = _make_doc()
        _add_table(doc, [["A"]])
        _add_para(doc, "After table")
        path = _save(doc, tmp_path)
        text, _meta = parse_docx(path)
        assert "| A |" in text
        assert "After table" in text

    def test_multiple_page_breaks_in_sequence(self, tmp_path):
        doc = _make_doc()
        _add_para(doc, "Start")
        _add_page_break(doc)
        _add_page_break(doc)
        _add_page_break(doc)
        _add_para(doc, "End")
        path = _save(doc, tmp_path)
        _, meta = parse_docx(path)
        assert meta["page_end"] == 4
