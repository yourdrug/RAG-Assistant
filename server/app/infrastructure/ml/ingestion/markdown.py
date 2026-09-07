"""Markdown parsing — section splitting, table detection, text cleaning."""

from __future__ import annotations

import html
import re
from pathlib import Path

from domain.value_objects.page_content_type import PageContentType

_MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_MD_TABLE_RE = re.compile(r"^(\|.+\|)\s*$", re.MULTILINE)
_DATE_IN_FILENAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _clean_markdown_text(text: str) -> str:
    """General markdown cleanup — used by both flat and section parsers."""
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"[image: \1](\2)", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`~]+", "", text)
    text = html.unescape(text)
    return text.strip()


def _flush_text_segment(current_text: list[str], segments: list[tuple[str, str]]) -> None:
    if current_text:
        text = "\n".join(current_text).strip()
        if text:
            segments.append((PageContentType.TEXT.value, text))
        current_text.clear()


def _flush_table_segment(current_table: list[str], segments: list[tuple[str, str]]) -> None:
    if current_table:
        table_text = "\n".join(current_table).strip()
        if table_text:
            segments.append((PageContentType.TABLE.value, table_text))
        current_table.clear()


def split_markdown_tables(content: str) -> list[tuple[str, str]]:
    """Split content into (content_type, text) segments.

    Detects markdown table blocks (lines starting and ending with |)
    and separates them from regular text. Returns list of (type, text)
    where type is 'text' or 'table'.
    """
    lines = content.split("\n")
    segments: list[tuple[str, str]] = []
    current_text: list[str] = []
    current_table: list[str] = []
    in_table = False

    for line in lines:
        is_table_line = bool(_MD_TABLE_RE.match(line.strip()))
        if is_table_line:
            if not in_table:
                _flush_text_segment(current_text, segments)
                in_table = True
            current_table.append(line)
        else:
            if in_table:
                _flush_table_segment(current_table, segments)
                in_table = False
            current_text.append(line)

    if in_table:
        _flush_table_segment(current_table, segments)
    else:
        _flush_text_segment(current_text, segments)

    return segments if segments else [(PageContentType.TEXT.value, content)]


def parse_markdown_sections(file_path: Path) -> list[tuple[str | None, str]]:
    r"""Split markdown into (heading, content) by header structure.

    Content between two headers belongs to the previous header. Text before
    the first header (if any) gets heading=None.

    Table blocks within sections are yielded as separate (heading, content) tuples
    with a special prefix '\\x00TABLE:' to signal content_type: table downstream.
    """
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    matches = list(_MD_HEADER_RE.finditer(raw))

    if not matches:
        segments = split_markdown_tables(_clean_markdown_text(raw))
        sections: list[tuple[str | None, str]] = []
        for ctype, text in segments:
            prefix = "\x00TABLE:" if ctype == PageContentType.TABLE.value else ""
            sections.append((None, prefix + text))
        return sections

    result: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        lead = _clean_markdown_text(raw[: matches[0].start()])
        if lead:
            for ctype, text in split_markdown_tables(lead):
                prefix = "\x00TABLE:" if ctype == PageContentType.TABLE.value else ""
                result.append((None, prefix + text))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        content = _clean_markdown_text(raw[start:end])
        if content:
            for ctype, text in split_markdown_tables(content):
                prefix = "\x00TABLE:" if ctype == PageContentType.TABLE.value else ""
                result.append((heading, prefix + text))

    return result


def parse_markdown(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return _clean_markdown_text(text)


def extract_date_from_filename(filename: str) -> str | None:
    """Find YYYY-MM-DD date in filename — simple heuristic for metadata."""
    m = _DATE_IN_FILENAME_RE.search(filename)
    return m.group(1) if m else None
