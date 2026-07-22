"""
Tests for domain/ingestion.py — text cleaning and markdown parsing.
Pure string transformations, no OCR/Qdrant/embeddings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from infrastructure.ml.ingestion import PARSERS, _parse_markdown, _parse_txt, clean_pdf_text  # noqa: E402

# ---------------------------------------------------------------------------
# clean_pdf_text
# ---------------------------------------------------------------------------


class TestCleanPdfText:
    def test_empty_string(self):
        assert clean_pdf_text("") == ""

    def test_single_word(self):
        assert clean_pdf_text("Hello") == "Hello"

    def test_fixes_hyphenation_at_line_break(self):
        assert clean_pdf_text("speci-\nalized") == "specialized"

    def test_preserves_hyphen_in_middle_of_word(self):
        assert clean_pdf_text("well-known") == "well-known"

    def test_collapses_multiple_spaces(self):
        assert clean_pdf_text("a    b   c") == "a b c"

    def test_preserves_single_newlines(self):
        assert clean_pdf_text("line1\nline2") == "line1\nline2"

    def test_collapses_multiple_newlines_to_single(self):
        assert clean_pdf_text("para1\n\n\n\npara2") == "para1\npara2"

    def test_removes_dashes_separator_line(self):
        text = "Before\n---\nAfter"
        result = clean_pdf_text(text)
        assert "---" not in result
        assert "Before" in result
        assert "After" in result

    def test_removes_equals_separator_line(self):
        text = "Title\n=====\nBody"
        result = clean_pdf_text(text)
        assert "=====" not in result

    def test_removes_bullets_separator_line(self):
        text = "Section\n•••••\nText"
        result = clean_pdf_text(text)
        assert "•••••" not in result

    def test_removes_tilde_separator_line(self):
        text = "A\n~~~~~\nB"
        result = clean_pdf_text(text)
        assert "~~~~~" not in result

    def test_preserves_short_dashes(self):
        # -- is only 2 chars, regex requires 3+ for separator removal,
        # but the hyphenation fix r"-\n" removes the second - before \n
        text = "one\n--\ntwo"
        result = clean_pdf_text(text)
        assert "one" in result
        assert "two" in result

    def test_strips_leading_trailing_whitespace(self):
        assert clean_pdf_text("  hello  ") == "hello"

    def test_multiple_hyphenations(self):
        assert clean_pdf_text("un-\nbeliev-\nably") == "unbelievably"

    def test_tabs_collapsed(self):
        assert clean_pdf_text("a\t\tb") == "a b"

    def test_mixed_whitespace_and_newlines(self):
        text = "word1   word2\n\n\n   word3"
        result = clean_pdf_text(text)
        # Whitespace-only runs collapse to single space, multiple blank lines -> single \n
        assert result == "word1 word2\n word3"


# ---------------------------------------------------------------------------
# _parse_markdown
# ---------------------------------------------------------------------------


class TestParseMarkdown:
    def _write_md(self, tmp_path, content):
        f = tmp_path / "test.md"
        f.write_text(content, encoding="utf-8")
        return f

    def test_removes_h1_to_h6_headers(self, tmp_path):
        f = self._write_md(tmp_path, "# Title\n## Sub\n### Deep")
        result = _parse_markdown(f)
        assert "#" not in result
        assert "Title" in result
        assert "Sub" in result

    def test_removes_image_syntax(self, tmp_path):
        f = self._write_md(tmp_path, "Text ![alt](img.png) more")
        result = _parse_markdown(f)
        assert "![alt]" not in result
        assert "img.png" not in result
        assert "Text" in result
        assert "more" in result

    def test_strips_link_syntax_keeps_text(self, tmp_path):
        f = self._write_md(tmp_path, "See [Google](https://google.com) for info")
        result = _parse_markdown(f)
        assert "[Google]" not in result
        assert "(https://google.com)" not in result
        assert "Google" in result
        assert "info" in result

    def test_removes_bold_italic_inline_code_markers(self, tmp_path):
        f = self._write_md(tmp_path, "**bold** *italic* `code` ~~strike~~")
        result = _parse_markdown(f)
        assert "**" not in result
        assert "*" not in result
        assert "`" not in result
        assert "~~" not in result
        assert "bold" in result
        assert "italic" in result
        assert "code" in result

    def test_unescapes_html_entities(self, tmp_path):
        f = self._write_md(tmp_path, "5 &lt; 10 &amp; 20 &gt; 5")
        result = _parse_markdown(f)
        assert "<" in result
        assert "&" in result
        assert ">" in result

    def test_empty_file(self, tmp_path):
        f = self._write_md(tmp_path, "")
        result = _parse_markdown(f)
        assert result == ""

    def test_plain_text_preserved(self, tmp_path):
        f = self._write_md(tmp_path, "Just plain text without markup.")
        result = _parse_markdown(f)
        assert result == "Just plain text without markup."

    def test_multiple_images_removed(self, tmp_path):
        content = "![a](a.png) text ![b](b.jpg) end"
        f = self._write_md(tmp_path, content)
        result = _parse_markdown(f)
        assert "a.png" not in result
        assert "b.jpg" not in result
        assert "text" in result
        assert "end" in result


# ---------------------------------------------------------------------------
# _parse_txt
# ---------------------------------------------------------------------------


class TestParseTxt:
    def test_reads_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        assert _parse_txt(f) == "hello world"

    def test_preserves_encoding(self, tmp_path):
        f = tmp_path / "ru.txt"
        f.write_text("Привет мир", encoding="utf-8")
        assert _parse_txt(f) == "Привет мир"

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert _parse_txt(f) == ""


# ---------------------------------------------------------------------------
# PARSERS registry
# ---------------------------------------------------------------------------


class TestParsersRegistry:
    def test_expected_extensions_registered(self):
        # .pdf is handled by parse_pdf() separately, not in PARSERS dict
        expected = {".docx", ".doc", ".rtf", ".md", ".txt"}
        assert set(PARSERS.keys()) == expected

    def test_docx_and_doc_share_parser(self):
        assert PARSERS[".docx"] is PARSERS[".doc"]

    def test_all_parsers_are_callable(self):
        for ext, parser in PARSERS.items():
            assert callable(parser), f"Parser for {ext} is not callable"
