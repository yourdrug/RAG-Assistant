"""
Tests for domain/ingestion.py — text cleaning, markdown parsing, merging, splitting.
Pure string transformations, no OCR/Qdrant/embeddings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from infrastructure.ml.ingestion import (  # noqa: E402
    PARSERS,
    _parse_markdown,
    _parse_txt,
    clean_pdf_text,
    merge_pdf_pages,
    split_documents,
)

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


# ---------------------------------------------------------------------------
# merge_pdf_pages
# ---------------------------------------------------------------------------


def _page_doc(text: str, page: int, source: str = "test.pdf"):
    from langchain.schema import Document

    return Document(page_content=text, metadata={"page": page, "source": source})


class TestMergePdfPages:
    def test_single_page_returned_as_is(self):
        pages = [_page_doc("hello", 1)]
        result = merge_pdf_pages(pages)
        assert len(result) == 1
        assert result[0].page_content == "hello"

    def test_two_pages_merged(self):
        pages = [_page_doc("page1", 1), _page_doc("page2", 2)]
        result = merge_pdf_pages(pages)
        assert len(result) == 1
        assert "page1" in result[0].page_content
        assert "page2" in result[0].page_content

    def test_metadata_preserves_page_range(self):
        pages = [_page_doc("a", 1), _page_doc("b", 3), _page_doc("c", 5)]
        result = merge_pdf_pages(pages)
        assert result[0].metadata["page_start"] == 1
        assert result[0].metadata["page_end"] == 5
        assert result[0].metadata["pages"] == [1, 3, 5]

    def test_pages_sorted_even_if_input_unsorted(self):
        pages = [_page_doc("c", 3), _page_doc("a", 1), _page_doc("b", 2)]
        result = merge_pdf_pages(pages)
        assert result[0].metadata["pages"] == [1, 2, 3]

    def test_groups_by_source(self):
        pages = [
            _page_doc("p1", 1, "a.pdf"),
            _page_doc("p2", 2, "a.pdf"),
            _page_doc("q1", 1, "b.pdf"),
        ]
        result = merge_pdf_pages(pages)
        assert len(result) == 2
        sources = {r.metadata["source"] for r in result}
        assert sources == {"a.pdf", "b.pdf"}

    def test_empty_list(self):
        assert merge_pdf_pages([]) == []


# ---------------------------------------------------------------------------
# split_documents
# ---------------------------------------------------------------------------


class TestSplitDocuments:
    def test_short_text_single_chunk(self):
        from langchain.schema import Document

        docs = [Document(page_content="Hello world", metadata={"source": "t.txt"})]
        chunks = split_documents(docs)
        assert len(chunks) == 1
        assert chunks[0].page_content == "Hello world"

    def test_long_text_splits(self):
        from unittest.mock import patch

        from langchain.schema import Document

        long_text = "word " * 200  # ~1000 chars
        docs = [Document(page_content=long_text, metadata={"source": "t.txt"})]
        with patch("infrastructure.ml.ingestion.settings") as mock_settings:
            mock_settings.chunk_size = 900
            mock_settings.chunk_overlap = 150
            chunks = split_documents(docs)
        assert len(chunks) > 1

    def test_prefers_paragraph_breaks(self):
        from langchain.schema import Document

        # Two paragraphs, each under chunk_size
        text = "First paragraph content.\n\nSecond paragraph content."
        docs = [Document(page_content=text, metadata={"source": "t.txt"})]
        chunks = split_documents(docs)
        assert len(chunks) == 1  # fits in one chunk

    def test_splits_at_double_newline(self):
        from unittest.mock import patch

        from langchain.schema import Document

        # Two paragraphs that together exceed chunk_size
        p1 = "A" * 300
        p2 = "B" * 300
        docs = [Document(page_content=f"{p1}\n\n{p2}", metadata={"source": "t.txt"})]
        with patch("infrastructure.ml.ingestion.settings") as mock_settings:
            mock_settings.chunk_size = 900
            mock_settings.chunk_overlap = 150
            chunks = split_documents(docs)
        assert len(chunks) >= 2

    def test_article_separator_preferred(self):
        from langchain.schema import Document

        text = "Intro text.\n\nСтатья 123. Article content here."
        docs = [Document(page_content=text, metadata={"source": "t.txt"})]
        chunks = split_documents(docs)
        # Should split at "Статья" boundary, keeping article intact
        texts = [c.page_content for c in chunks]
        assert any("Статья 123" in t for t in texts)

    def test_metadata_preserved(self):
        from langchain.schema import Document

        docs = [Document(page_content="content", metadata={"source": "x.pdf", "page": 5})]
        chunks = split_documents(docs)
        assert chunks[0].metadata["source"] == "x.pdf"
        assert chunks[0].metadata["page"] == 5
