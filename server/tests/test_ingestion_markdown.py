"""Tests for infrastructure/ml/ingestion/markdown.py — section splitting, setext
normalization, code fence masking, date extraction, and doc title extraction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from infrastructure.ml.ingestion.markdown import (  # noqa: E402
    _mask_code_fences,
    _normalize_setext_headers,
    _unmask_code_fences,
    extract_date_from_filename,
    extract_doc_title,
    parse_markdown_sections,
)


# ---------------------------------------------------------------------------
# _mask_code_fences
# ---------------------------------------------------------------------------


class TestMaskCodeFences:
    def test_replaces_fenced_block(self):
        raw = "text\n```python\nx = 1\n```\nmore"
        masked, blocks = _mask_code_fences(raw)
        assert len(blocks) == 1
        assert "```python" not in masked
        assert "x = 1" not in masked

    def test_preserves_line_count(self):
        raw = "line1\n```\nline2\nline3\n```\nline4"
        masked, _blocks = _mask_code_fences(raw)
        assert masked.count("\n") == raw.count("\n")

    def test_multiple_fences(self):
        raw = "a\n```\nfoo\n```\nb\n```\nbar\n```\nc"
        masked, blocks = _mask_code_fences(raw)
        assert len(blocks) == 2
        assert "foo" not in masked
        assert "bar" not in masked

    def test_unmask_restores_block_content(self):
        raw = "before\n```\ncode here\n```\nafter"
        masked, blocks = _mask_code_fences(raw)
        restored = _unmask_code_fences(masked, blocks)
        # The original fenced block is fully restored inside the text
        assert blocks[0] in restored
        assert "code here" in restored

    def test_no_fences_unchanged(self):
        raw = "just plain text\nno fences"
        masked, blocks = _mask_code_fences(raw)
        assert masked == raw
        assert blocks == []

    def test_tilde_fences(self):
        raw = "text\n~~~\nblock\n~~~\nend"
        masked, blocks = _mask_code_fences(raw)
        assert len(blocks) == 1
        assert "block" not in masked


# ---------------------------------------------------------------------------
# _normalize_setext_headers
# ---------------------------------------------------------------------------


class TestNormalizeSetextHeaders:
    def test_converts_equals_to_h1(self):
        raw = "Title\n==="
        result = _normalize_setext_headers(raw)
        assert result.startswith("# Title")
        assert "===" not in result

    def test_converts_dashes_to_h2(self):
        raw = "Subtitle\n---"
        result = _normalize_setext_headers(raw)
        assert result.startswith("## Subtitle")
        assert "---" not in result

    def test_does_not_touch_atx_headers(self):
        raw = "# ATX Title"
        result = _normalize_setext_headers(raw)
        assert result == raw

    def test_does_not_touch_code_blocks(self):
        raw = "```\nTitle\n===\n```"
        result = _normalize_setext_headers(raw)
        assert "===" in result

    def test_does_not_touch_table_separators(self):
        raw = "| col |\n|-----|"
        result = _normalize_setext_headers(raw)
        assert "|-----|" in result


# ---------------------------------------------------------------------------
# extract_date_from_filename
# ---------------------------------------------------------------------------


class TestExtractDateFromFilename:
    def test_extracts_date(self):
        assert extract_date_from_filename("report-2024-03-15.pdf") == "2024-03-15"

    def test_returns_none_when_no_date(self):
        assert extract_date_from_filename("report.pdf") is None

    def test_returns_first_date_if_multiple(self):
        assert extract_date_from_filename("2023-01-01_to_2024-06-30.pdf") == "2023-01-01"


# ---------------------------------------------------------------------------
# extract_doc_title
# ---------------------------------------------------------------------------


class TestExtractDocTitle:
    def _write_md(self, tmp_path, content, name="doc.md"):
        f = tmp_path / name
        f.write_text(content, encoding="utf-8")
        return f

    def test_returns_first_h1(self, tmp_path):
        f = self._write_md(tmp_path, "# My Title\n\nSome content.\n## Sub")
        assert extract_doc_title(f) == "My Title"

    def test_returns_first_h2_if_no_h1(self, tmp_path):
        f = self._write_md(tmp_path, "## Second Level\nContent")
        assert extract_doc_title(f) == "Second Level"

    def test_returns_none_for_no_headers(self, tmp_path):
        f = self._write_md(tmp_path, "Just some text.\nNo headers.")
        assert extract_doc_title(f) is None

    def test_skips_code_fences(self, tmp_path):
        f = self._write_md(tmp_path, "```\n# Not a header\n```\n# Real Title")
        assert extract_doc_title(f) == "Real Title"


# ---------------------------------------------------------------------------
# parse_markdown_sections
# ---------------------------------------------------------------------------


class TestParseMarkdownSections:
    def _write_md(self, tmp_path, content, name="doc.md"):
        f = tmp_path / name
        f.write_text(content, encoding="utf-8")
        return f

    def test_splits_by_atx_headers(self, tmp_path):
        content = "# H1\nBody1\n## H2\nBody2"
        f = self._write_md(tmp_path, content)
        sections = parse_markdown_sections(f)
        headings = [h for h, _ in sections]
        assert "H1" in headings[0]
        assert "H1 > H2" in headings[1]

    def test_splits_by_setext_headers(self, tmp_path):
        content = "Title One\n===\nBody1\nTitle Two\n---\nBody2"
        f = self._write_md(tmp_path, content)
        sections = parse_markdown_sections(f)
        assert len(sections) >= 2
        headings = [h for h, _ in sections]
        assert "Title One" in headings[0]
        assert "Title Two" in headings[1]

    def test_breadcrumb_path_built(self, tmp_path):
        content = "# H1\nBody1\n## H2\nBody2\n### H3\nBody3"
        f = self._write_md(tmp_path, content)
        sections = parse_markdown_sections(f)
        assert len(sections) >= 3
        # Deepest section has full breadcrumb
        assert "H1 > H2 > H3" in sections[2][0]
        # Intermediate section has partial breadcrumb
        assert sections[1][0] == "H1 > H2"

    def test_leading_text_before_first_header(self, tmp_path):
        content = "Intro text.\n# First Header\nBody"
        f = self._write_md(tmp_path, content)
        sections = parse_markdown_sections(f)
        # First section should have heading=None for leading text
        assert sections[0][0] is None
        assert "Intro text" in sections[0][1]

    def test_code_blocks_not_treated_as_headers(self, tmp_path):
        content = "# Real Header\nBody\n```\n# Fake Header\n```\nMore"
        f = self._write_md(tmp_path, content)
        sections = parse_markdown_sections(f)
        headings = [h for h, _ in sections if h is not None]
        assert len(headings) == 1
        assert "Real Header" in headings[0]

    def test_tables_yielded_as_table_sections(self, tmp_path):
        content = "# Section\nText\n| A | B |\n|---|---|\n| 1 | 2 |"
        f = self._write_md(tmp_path, content)
        sections = parse_markdown_sections(f)
        types = []
        for _, text in sections:
            if text.startswith("\x00TABLE:"):
                types.append("table")
            else:
                types.append("text")
        assert "table" in types
        assert "text" in types

    def test_fenced_code_preserved_in_content(self, tmp_path):
        content = "# Header\n```python\nprint('hello')\n```"
        f = self._write_md(tmp_path, content)
        sections = parse_markdown_sections(f)
        all_content = " ".join(t for _, t in sections)
        assert "print('hello')" in all_content

    def test_no_headers_returns_single_section(self, tmp_path):
        content = "Just some text.\nNo headers here."
        f = self._write_md(tmp_path, content)
        sections = parse_markdown_sections(f)
        assert len(sections) == 1
        assert sections[0][0] is None
        assert "Just some text" in sections[0][1]
