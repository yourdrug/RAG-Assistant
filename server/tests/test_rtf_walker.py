"""Characterization tests for ml/ingestion/rtf.py — _walk_paragraphs and helpers.

Locks down current behavior BEFORE decomposing the 234-line _walk_paragraphs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from infrastructure.ml.ingestion.rtf import (  # noqa: E402
    _Paragraph,
    _Table,
    _walk_paragraphs,
    extract_doc_title,
    extract_rtf_tables,
    parse_rtf,
    parse_rtf_sections,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rtf(text: str) -> str:
    """Wrap text in minimal RTF envelope."""
    return r"{\rtf1\ansi " + text + "}"


def _paragraphs(rtf: str) -> list[_Paragraph]:
    """Extract only _Paragraph segments from _walk_paragraphs output."""
    return [s.paragraph for s in _walk_paragraphs(rtf) if s.paragraph is not None]


def _tables(rtf: str) -> list[_Table]:
    """Extract only _Table segments from _walk_paragraphs output."""
    return [s.table for s in _walk_paragraphs(rtf) if s.table is not None]


# ---------------------------------------------------------------------------
# Basic paragraph extraction
# ---------------------------------------------------------------------------


class TestWalkParagraphsBasic:
    def test_single_paragraph(self):
        segs = _walk_paragraphs(_rtf("Hello world"))
        paras = [s for s in segs if s.paragraph]
        assert len(paras) == 1
        assert paras[0].paragraph.text == "Hello world"

    def test_multiple_paragraphs(self):
        rtf = _rtf(r"First paragraph\par Second paragraph\par Third")
        paras = _paragraphs(rtf)
        assert len(paras) == 3
        assert paras[0].text == "First paragraph"
        assert paras[1].text == "Second paragraph"
        assert paras[2].text == "Third"

    def test_paragraph_with_newlines_in_rtf(self):
        rtf = _rtf("Line one\nLine two")
        paras = _paragraphs(rtf)
        assert len(paras) == 1
        assert "Line one" in paras[0].text
        assert "Line two" in paras[0].text

    def test_empty_rtf(self):
        segs = _walk_paragraphs(_rtf(""))
        assert len(segs) == 0

    def test_whitespace_only(self):
        segs = _walk_paragraphs(_rtf("   \t  "))
        paras = [s.paragraph for s in segs if s.paragraph]
        # Whitespace-only paragraphs are filtered out
        assert len(paras) == 0


# ---------------------------------------------------------------------------
# Bold detection
# ---------------------------------------------------------------------------


class TestWalkParagraphsBold:
    def test_bold_paragraph(self):
        rtf = _rtf(r"\b Bold text\b0")
        paras = _paragraphs(rtf)
        assert len(paras) == 1
        assert paras[0].bold is True

    def test_non_bold_paragraph(self):
        rtf = _rtf("Plain text")
        paras = _paragraphs(rtf)
        assert len(paras) == 1
        assert paras[0].bold is False

    def test_mixed_bold_non_bold(self):
        # \b turns on bold, \par flushes but doesn't reset — "Plain" inherits bold
        rtf = _rtf(r"\b Bold\pard\par Plain")
        paras = _paragraphs(rtf)
        assert len(paras) == 2
        assert paras[0].bold is True
        assert paras[1].bold is False


# ---------------------------------------------------------------------------
# Font size detection
# ---------------------------------------------------------------------------


class TestWalkParagraphsFontSize:
    def test_font_size_set(self):
        rtf = _rtf(r"\fs28 Large text")
        paras = _paragraphs(rtf)
        assert len(paras) == 1
        assert paras[0].font_size == 14.0  # fs28 = 14 points

    def test_font_size_not_set(self):
        rtf = _rtf("Plain text")
        paras = _paragraphs(rtf)
        assert paras[0].font_size is None

    def test_font_size_inherited(self):
        rtf = _rtf(r"{\fs28 Bold heading\par Normal text}")
        paras = _paragraphs(rtf)
        assert len(paras) == 2
        # First paragraph inherits fs28 from parent group
        assert paras[0].font_size == 14.0


# ---------------------------------------------------------------------------
# Centering detection
# ---------------------------------------------------------------------------


class TestWalkParagraphsCentering:
    def test_centered_paragraph(self):
        rtf = _rtf(r"\qc Centered text")
        paras = _paragraphs(rtf)
        assert len(paras) == 1
        assert paras[0].centered is True

    def test_centering_reset(self):
        rtf = _rtf(r"\qc Centered\par ql Not centered")
        paras = _paragraphs(rtf)
        assert paras[0].centered is True
        assert paras[1].centered is False


# ---------------------------------------------------------------------------
# Group nesting
# ---------------------------------------------------------------------------


class TestWalkParagraphsGroups:
    def test_nested_groups_preserve_state(self):
        # Bold group wraps everything — all chars are bold
        rtf = _rtf(r"{\b Bold text here}")
        paras = _paragraphs(rtf)
        assert len(paras) == 1
        assert paras[0].bold is True

    def test_skip_destination(self):
        rtf = _rtf(r"{\fonttbl{\f0 Times New Roman;}} Real text")
        paras = _paragraphs(rtf)
        assert len(paras) == 1
        assert "fonttbl" not in paras[0].text
        assert "Real text" in paras[0].text

    def test_group_pop_restores_state(self):
        # \b inside group, then group closes, \par separates, "Normal" is not bold
        rtf = _rtf(r"{\b Bold}\par Normal")
        paras = _paragraphs(rtf)
        assert len(paras) == 2
        assert paras[0].bold is True
        assert paras[1].bold is False


# ---------------------------------------------------------------------------
# Hex escapes
# ---------------------------------------------------------------------------


class TestWalkParagraphsHexEscape:
    def test_hex_escape_decoded(self):
        rtf = _rtf(r"Hello\'20World")
        paras = _paragraphs(rtf)
        assert len(paras) == 1
        assert "Hello World" in paras[0].text

    def test_hex_escape_cyrillic(self):
        # \'cf\'e5\'f2 in cp1251 = "Пет"
        rtf = _rtf(r"\'cf\'e5\'f2")
        paras = _paragraphs(rtf)
        assert len(paras) == 1
        assert "Пет" in paras[0].text


# ---------------------------------------------------------------------------
# Special symbols
# ---------------------------------------------------------------------------


class TestWalkParagraphsSpecialSymbols:
    def test_emdash(self):
        rtf = _rtf(r"Before\emdash After")
        paras = _paragraphs(rtf)
        assert "-" in paras[0].text

    def test_endash(self):
        rtf = _rtf(r"Before\endash After")
        paras = _paragraphs(rtf)
        assert "-" in paras[0].text

    def test_nbsp(self):
        rtf = _rtf(r"Before\~After")
        paras = _paragraphs(rtf)
        assert "\u00a0" in paras[0].text

    def test_nonbreaking_hyphen(self):
        rtf = _rtf(r"Before" + "\\" + "_After")
        paras = _paragraphs(rtf)
        assert "-" in paras[0].text


# ---------------------------------------------------------------------------
# Table detection
# ---------------------------------------------------------------------------


class TestWalkParagraphsTables:
    def test_simple_table(self):
        rtf = _rtf(r"\intbl Cell1\cell Cell2\cell\row")
        tables = _tables(rtf)
        assert len(tables) == 1
        assert len(tables[0].rows) == 1
        assert tables[0].rows[0] == ["Cell1", "Cell2"]

    def test_multi_row_table(self):
        rtf = _rtf(r"\intbl A\cell B\cell\row \intbl C\cell D\cell\row")
        tables = _tables(rtf)
        assert len(tables) == 1
        assert len(tables[0].rows) == 2

    def test_table_with_paragraphs(self):
        rtf = _rtf(r"Before\par \intbl Cell\cell\row After")
        segs = _walk_paragraphs(rtf)
        paras = [s.paragraph for s in segs if s.paragraph]
        tables = [s.table for s in segs if s.table]
        assert len(tables) == 1
        # "Before" is a paragraph, table, "After" is a paragraph
        assert any("Before" in p.text for p in paras)
        assert any("After" in p.text for p in paras)


# ---------------------------------------------------------------------------
# sect/page control words
# ---------------------------------------------------------------------------


class TestWalkParagraphsControlWords:
    def test_sect_break(self):
        rtf = _rtf(r"Para1\sect Para2")
        paras = _paragraphs(rtf)
        assert len(paras) == 2

    def test_page_break(self):
        rtf = _rtf(r"Para1\page Para2")
        paras = _paragraphs(rtf)
        assert len(paras) == 2

    def test_tab_character(self):
        rtf = _rtf(r"Col1\tab Col2")
        paras = _paragraphs(rtf)
        # \tab inserts tab but flush_paragraph normalizes whitespace to space
        assert "Col1" in paras[0].text
        assert "Col2" in paras[0].text


# ---------------------------------------------------------------------------
# Existing public API
# ---------------------------------------------------------------------------


class TestParseRtf:
    def test_extracts_text(self, tmp_path):
        f = tmp_path / "test.rtf"
        f.write_text(_rtf("Hello world"), encoding="utf-8")
        text, meta = parse_rtf(f)
        assert "Hello world" in text

    def test_extracts_codepage(self, tmp_path):
        f = tmp_path / "test.rtf"
        f.write_text(_rtf("text"), encoding="utf-8")
        _, meta = parse_rtf(f)
        assert "codepage" in meta


class TestExtractDocTitle:
    def test_no_title(self, tmp_path):
        f = tmp_path / "test.rtf"
        f.write_text(_rtf("No title"), encoding="utf-8")
        assert extract_doc_title(f) is None


class TestParseRtfSections:
    def test_plain_text_returns_single_section(self, tmp_path):
        f = tmp_path / "test.rtf"
        f.write_text(_rtf("Just text"), encoding="utf-8")
        sections = parse_rtf_sections(f)
        assert len(sections) == 1
        assert sections[0][0] is None  # no heading


class TestExtractRtfTables:
    def test_extracts_tables(self):
        rtf = _rtf(r"\intbl A\cell B\cell\row")
        tables = extract_rtf_tables(rtf)
        assert len(tables) == 1
        assert "A" in tables[0]
        assert "B" in tables[0]
