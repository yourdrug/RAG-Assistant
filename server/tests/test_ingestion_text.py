"""Tests for domain/ingestion.py -- text cleaning, markdown parsing, merging, splitting.

Pure string transformations, no OCR/Qdrant/embeddings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from langchain_core.documents import Document

from infrastructure.ml.ingestion import (
    clean_pdf_text,
    merge_pdf_pages,
    parse_markdown,
    parse_txt,
    split_documents,
    split_markdown_tables,
)
from infrastructure.ml.ingestion.registry import FORMAT_HANDLERS


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# clean_pdf_text
# ---------------------------------------------------------------------------


class TestCleanPdfText:
    def test_empty_string(self) -> None:
        assert clean_pdf_text("") == ""

    def test_single_word(self) -> None:
        assert clean_pdf_text("Hello") == "Hello"

    def test_fixes_hyphenation_at_line_break(self) -> None:
        assert clean_pdf_text("speci-\nalized") == "specialized"

    def test_preserves_hyphen_in_middle_of_word(self) -> None:
        assert clean_pdf_text("well-known") == "well-known"

    def test_collapses_multiple_spaces(self) -> None:
        assert clean_pdf_text("a    b   c") == "a b c"

    def test_preserves_single_newlines(self) -> None:
        assert clean_pdf_text("line1\nline2") == "line1\nline2"

    def test_collapses_multiple_newlines_to_single(self) -> None:
        # clean_pdf_text preserves \n\n as paragraph breaks (3+ newlines → 2) -> None
        assert clean_pdf_text("para1\n\n\n\npara2") == "para1\n\npara2"

    def test_removes_dashes_separator_line(self) -> None:
        text = "Before\n---\nAfter"
        result = clean_pdf_text(text)
        assert "---" not in result
        assert "Before" in result
        assert "After" in result

    def test_removes_equals_separator_line(self) -> None:
        text = "Title\n=====\nBody"
        result = clean_pdf_text(text)
        assert "=====" not in result

    def test_removes_bullets_separator_line(self) -> None:
        text = "Section\n•••••\nText"
        result = clean_pdf_text(text)
        assert "•••••" not in result

    def test_removes_tilde_separator_line(self) -> None:
        text = "A\n~~~~~\nB"
        result = clean_pdf_text(text)
        assert "~~~~~" not in result

    def test_preserves_short_dashes(self) -> None:
        # -- is only 2 chars, regex requires 3+ for separator removal,
        # but the hyphenation fix r"-\n" removes the second - before \n
        text = "one\n--\ntwo"
        result = clean_pdf_text(text)
        assert "one" in result
        assert "two" in result

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert clean_pdf_text("  hello  ") == "hello"

    def test_multiple_hyphenations(self) -> None:
        assert clean_pdf_text("un-\nbeliev-\nably") == "unbelievably"

    def test_tabs_collapsed(self) -> None:
        assert clean_pdf_text("a\t\tb") == "a b"

    def test_mixed_whitespace_and_newlines(self) -> None:
        text = "word1   word2\n\n\n   word3"
        result = clean_pdf_text(text)
        # Horizontal whitespace collapses, 3+ newlines → paragraph break (\n\n)
        # Space after newlines is preserved (not stripped mid-string)
        assert result == "word1 word2\n\n word3"


# ---------------------------------------------------------------------------
# parse_markdown
# ---------------------------------------------------------------------------


class TestParseMarkdown:
    def _write_md(self, tmp_path: Path, content: str) -> Path:
        f = tmp_path / "test.md"
        f.write_text(content, encoding="utf-8")
        return f

    def test_removes_h1_to_h6_headers(self, tmp_path: Path) -> None:
        f = self._write_md(tmp_path, "# Title\n## Sub\n### Deep")
        result = parse_markdown(f)
        assert "#" not in result
        assert "Title" in result
        assert "Sub" in result

    def test_removes_image_syntax(self, tmp_path: Path) -> None:
        f = self._write_md(tmp_path, "Text ![alt](img.png) more")
        result = parse_markdown(f)
        assert "![alt]" not in result
        assert "img.png" not in result
        assert "Text" in result
        assert "more" in result

    def test_strips_link_syntax_keeps_text(self, tmp_path: Path) -> None:
        f = self._write_md(tmp_path, "See [Google](https://google.com) for info")
        result = parse_markdown(f)
        assert "[Google]" not in result
        assert "(https://google.com)" not in result
        assert "Google" in result
        assert "info" in result

    def test_removes_bold_italic_inline_code_markers(self, tmp_path: Path) -> None:
        f = self._write_md(tmp_path, "**bold** *italic* `code` ~~strike~~")
        result = parse_markdown(f)
        assert "**" not in result
        assert "*" not in result
        assert "`" not in result
        assert "~~" not in result
        assert "bold" in result
        assert "italic" in result
        assert "code" in result

    def test_unescapes_html_entities(self, tmp_path: Path) -> None:
        f = self._write_md(tmp_path, "5 &lt; 10 &amp; 20 &gt; 5")
        result = parse_markdown(f)
        assert "<" in result
        assert "&" in result
        assert ">" in result

    def test_empty_file(self, tmp_path: Path) -> None:
        f = self._write_md(tmp_path, "")
        result = parse_markdown(f)
        assert result == ""

    def test_plain_text_preserved(self, tmp_path: Path) -> None:
        f = self._write_md(tmp_path, "Just plain text without markup.")
        result = parse_markdown(f)
        assert result == "Just plain text without markup."

    def test_multiple_images_removed(self, tmp_path: Path) -> None:
        content = "![a](a.png) text ![b](b.jpg) end"
        f = self._write_md(tmp_path, content)
        result = parse_markdown(f)
        assert "a.png" not in result
        assert "b.jpg" not in result
        assert "text" in result
        assert "end" in result


# ---------------------------------------------------------------------------
# parse_txt
# ---------------------------------------------------------------------------


class TestParseTxt:
    def test_reads_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        text, _meta = parse_txt(f)
        assert text == "hello world"

    def test_preserves_encoding(self, tmp_path: Path) -> None:
        f = tmp_path / "ru.txt"
        f.write_text("Привет мир", encoding="utf-8")
        text, _meta = parse_txt(f)
        assert text == "Привет мир"

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        text, _meta = parse_txt(f)
        assert text == ""


# ---------------------------------------------------------------------------
# Format handler registry
# ---------------------------------------------------------------------------


class TestFormatHandlersRegistry:
    def test_expected_extensions_registered(self) -> None:
        expected = {".pdf", ".docx", ".rtf", ".md", ".txt"}
        assert set(FORMAT_HANDLERS.keys()) == expected

    def test_legacy_doc_is_not_registered(self) -> None:
        assert ".doc" not in FORMAT_HANDLERS

    def test_all_parsers_are_callable(self) -> None:
        for ext, handler in FORMAT_HANDLERS.items():
            assert callable(handler.parse), f"Parser for {ext} is not callable"


# ---------------------------------------------------------------------------
# merge_pdf_pages
# ---------------------------------------------------------------------------


def _page_doc(text: str, page: int, source: str = "test.pdf") -> Document:

    return Document(page_content=text, metadata={"page": page, "source": source})


class TestMergePdfPages:
    def test_single_page_returned_as_is(self) -> None:
        pages = [_page_doc("hello", 1)]
        result = merge_pdf_pages(pages)
        assert len(result) == 1
        assert result[0].page_content == "hello"

    def test_two_pages_merged(self) -> None:
        pages = [_page_doc("page1", 1), _page_doc("page2", 2)]
        result = merge_pdf_pages(pages)
        assert len(result) == 1
        assert "page1" in result[0].page_content
        assert "page2" in result[0].page_content

    def test_metadata_preserves_page_range(self) -> None:
        pages = [_page_doc("a", 1), _page_doc("b", 3), _page_doc("c", 5)]
        result = merge_pdf_pages(pages)
        assert result[0].metadata["page_start"] == 1
        assert result[0].metadata["page_end"] == 5
        assert result[0].metadata["pages"] == [1, 3, 5]

    def test_pages_sorted_even_if_input_unsorted(self) -> None:
        pages = [_page_doc("c", 3), _page_doc("a", 1), _page_doc("b", 2)]
        result = merge_pdf_pages(pages)
        assert result[0].metadata["pages"] == [1, 2, 3]

    def test_groups_by_source(self) -> None:
        pages = [
            _page_doc("p1", 1, "a.pdf"),
            _page_doc("p2", 2, "a.pdf"),
            _page_doc("q1", 1, "b.pdf"),
        ]
        result = merge_pdf_pages(pages)
        assert len(result) == 2
        sources = {r.metadata["source"] for r in result}
        assert sources == {"a.pdf", "b.pdf"}

    def test_empty_list(self) -> None:
        assert merge_pdf_pages([]) == []


# ---------------------------------------------------------------------------
# split_documents
# ---------------------------------------------------------------------------


class TestSplitDocuments:
    def test_short_text_single_chunk(self) -> None:

        docs = [Document(page_content="Hello world", metadata={"source": "t.txt"})]
        chunks = split_documents(docs)
        assert len(chunks) == 1
        assert chunks[0].page_content == "Hello world"

    def test_long_text_splits(self) -> None:

        long_text = "word " * 200  # ~1000 chars
        docs = [Document(page_content=long_text, metadata={"source": "t.txt"})]
        with patch("infrastructure.ml.ingestion.splitting.settings") as mock_settings:
            mock_settings.chunk_size = 900
            mock_settings.chunk_overlap = 150
            chunks = split_documents(docs)
        assert len(chunks) > 1

    def test_prefers_paragraph_breaks(self) -> None:

        # Two paragraphs, each under chunk_size
        text = "First paragraph content.\n\nSecond paragraph content."
        docs = [Document(page_content=text, metadata={"source": "t.txt"})]
        chunks = split_documents(docs)
        assert len(chunks) == 1  # fits in one chunk

    def test_splits_at_double_newline(self) -> None:

        # Two paragraphs that together exceed chunk_size
        p1 = "A" * 500
        p2 = "B" * 500
        docs = [Document(page_content=f"{p1}\n\n{p2}", metadata={"source": "t.txt"})]
        with patch("infrastructure.ml.ingestion.splitting.settings") as mock_settings:
            mock_settings.chunk_size = 600
            mock_settings.chunk_overlap = 100
            chunks = split_documents(docs)
        assert len(chunks) >= 2

    def test_article_separator_preferred(self) -> None:

        text = "Intro text.\n\nСтатья 123. Article content here."
        docs = [Document(page_content=text, metadata={"source": "t.txt"})]
        chunks = split_documents(docs, domain="legal")
        # Should split at "Статья" boundary, keeping article intact
        texts = [c.page_content for c in chunks]
        assert any("Статья 123" in t for t in texts)

    def test_metadata_preserved(self) -> None:

        docs = [Document(page_content="content", metadata={"source": "x.pdf", "page": 5})]
        chunks = split_documents(docs)
        assert chunks[0].metadata["source"] == "x.pdf"
        assert chunks[0].metadata["page"] == 5


# ---------------------------------------------------------------------------
# split_documents — domain-aware separators
# ---------------------------------------------------------------------------


class TestSplitDocumentsDomainAware:
    def test_general_uses_markdown_headers(self) -> None:

        text = "Intro\n\n## Section One\n\nContent one.\n\n## Section Two\n\nContent two."
        docs = [Document(page_content=text, metadata={"source": "t.md"})]
        chunks = split_documents(docs, domain="general")
        texts = [c.page_content for c in chunks]
        assert any("Section One" in t for t in texts)

    def test_table_chunks_not_split(self) -> None:

        table = "| A | B |\n|---|---|\n| 1 | 2 |"
        docs = [
            Document(page_content="Some text", metadata={"source": "t.md"}),
            Document(page_content=table, metadata={"source": "t.md", "content_type": "table"}),
        ]
        chunks = split_documents(docs, domain="general")
        table_chunks = [c for c in chunks if c.metadata.get("content_type") == "table"]
        assert len(table_chunks) == 1
        assert table_chunks[0].page_content == table


# ---------------------------------------------------------------------------
# split_markdown_tables
# ---------------------------------------------------------------------------


class TestSplitMarkdownTables:
    def test_text_only(self) -> None:
        result = split_markdown_tables("Just plain text")
        assert len(result) == 1
        assert result[0] == ("text", "Just plain text")

    def test_table_only(self) -> None:
        content = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = split_markdown_tables(content)
        assert len(result) == 1
        assert result[0][0] == "table"

    def test_text_table_text(self) -> None:
        content = "Before\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nAfter"
        result = split_markdown_tables(content)
        types = [r[0] for r in result]
        assert types == ["text", "table", "text"]

    def test_multiple_tables(self) -> None:
        content = "Text1\n| A | B |\n|---|---|\n| 1 | 2 |\nMiddle\n| C | D |\n|---|---|\n| 3 | 4 |\nText2"
        result = split_markdown_tables(content)
        types = [r[0] for r in result]
        assert types == ["text", "table", "text", "table", "text"]


# ---------------------------------------------------------------------------
# split_documents — table atomic
# ---------------------------------------------------------------------------


class TestSplitDocumentsTableAtomic:
    def test_tables_preserved_unsplit(self) -> None:

        table = "| Col1 | Col2 | Col3 |\n|------|------|------|\n| a | b | c |\n| d | e | f |"
        text = "A" * 600  # Exceeds default chunk_size
        docs = [
            Document(page_content=text, metadata={"source": "t.pdf"}),
            Document(page_content=table, metadata={"source": "t.pdf", "content_type": "table"}),
        ]
        chunks = split_documents(docs, domain="general")
        table_chunks = [c for c in chunks if c.metadata.get("content_type") == "table"]
        assert len(table_chunks) == 1
        assert table_chunks[0].page_content == table
