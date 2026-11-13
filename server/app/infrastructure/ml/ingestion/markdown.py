"""Markdown parsing — section splitting, table detection, text cleaning."""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

from domain.ingestion.document_tree import NodeKind
from domain.knowledge.enums import PageContentType
from infrastructure.ml.ingestion.format_tree import DocumentTreeBuilder, sections_from_tree
from infrastructure.ml.ingestion.sections import table_context_sections


if TYPE_CHECKING:
    from pathlib import Path

    from domain.ingestion.document_tree import DocNode
    from infrastructure.ml.ingestion.section_types import Section


_MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
# Setext headers: a line of text followed immediately by a line of only
# "=" (h1) or "-" (h2). Must not contain "|" so we never mistake a table
# separator row ("---|---") for a setext underline.
_MD_SETEXT_RE = re.compile(r"^([^\n|]+?)[ \t]*\n(=+|-{2,})[ \t]*$", re.MULTILINE)
_MD_TABLE_RE = re.compile(r"^(\|.+\|)\s*$", re.MULTILINE)
_MD_FENCE_BLOCK_RE = re.compile(r"^([`~]{3,}).*?\n.*?^\1[`~]*\s*$", re.MULTILINE | re.DOTALL)
_DATE_IN_FILENAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})")
_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HASHTAG_RE = re.compile(r"(?:^|\s)#([a-zA-Zа-яА-ЯёЁ0-9_-]+)", re.MULTILINE)

_FENCE_PLACEHOLDER = "\x00CODEFENCE{}\x00"


def _mask_code_fences(raw: str) -> tuple[str, list[str]]:
    """Replace fenced code blocks with opaque placeholders.

    Prevents '#' comments or '|' characters inside code from being
    misdetected as markdown headers or table rows. The placeholder line
    count matches the original so header/table byte offsets used for
    section slicing stay valid.
    """
    blocks: list[str] = []

    def _replace(m: re.Match[str]) -> str:
        original = m.group(0)
        blocks.append(original)
        newline_count = original.count("\n")
        placeholder = _FENCE_PLACEHOLDER.format(len(blocks) - 1)
        return placeholder + "\n" * newline_count

    masked = _MD_FENCE_BLOCK_RE.sub(_replace, raw)
    return masked, blocks


def _unmask_code_fences(text: str, blocks: list[str]) -> str:
    for idx, block in enumerate(blocks):
        text = text.replace(_FENCE_PLACEHOLDER.format(idx), block)

    return text


def _normalize_setext_headers(raw: str) -> str:
    r"""Rewrite setext-style headers ("Title\\n===") into ATX form ("# Title").

    Lets the rest of the pipeline reason about a single header syntax.
    Applied on code-fence-masked text so underlines inside code blocks are
    never touched.
    """
    masked, blocks = _mask_code_fences(raw)

    def _replace(m: re.Match[str]) -> str:
        title = m.group(1).strip()
        underline = m.group(2)
        level = "#" if underline.startswith("=") else "##"
        return f"{level} {title}"

    masked = _MD_SETEXT_RE.sub(_replace, masked)
    return _unmask_code_fences(masked, blocks)


def _clean_markdown_text(text: str) -> str:
    """General markdown cleanup — used by both flat and section parsers."""
    masked, blocks = _mask_code_fences(text)
    masked = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"[image: \1](\2)", masked)
    masked = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", masked)
    masked = re.sub(r"^#{1,6}\s+", "", masked, flags=re.MULTILINE)
    masked = re.sub(r"[*_`~]+", "", masked)
    text = _unmask_code_fences(masked, blocks)
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
    and separates them from regular text, ignoring anything inside fenced
    code blocks. Returns list of (type, text) where type is 'text' or 'table'.
    """
    masked, blocks = _mask_code_fences(content)
    lines = masked.split("\n")
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

    segments = [(ctype, _unmask_code_fences(text, blocks)) for ctype, text in segments]
    return segments or [(PageContentType.TEXT.value, content)]


def parse_markdown_tree(file_path: Path) -> DocNode:
    r"""Build the markdown tree from the header hierarchy.

    Content between two headers belongs to the previous header. Text before
    the first header (if any) stays at the document root. Both ATX ("# Title")
    and setext ("Title\\n===") headers are recognized, and headers/tables inside
    fenced code blocks are ignored. Header levels nest the heading nodes, so a
    deeply nested section's chunks keep their full path for retrieval.

    Table blocks are separate nodes with an explicit table kind.
    """
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    raw = _normalize_setext_headers(raw)

    masked, fence_blocks = _mask_code_fences(raw)
    matches = list(_MD_HEADER_RE.finditer(masked))
    builder = DocumentTreeBuilder()

    if not matches:
        add_markdown_blocks(builder, split_markdown_tables(_clean_markdown_text(raw)))
        return builder.build()

    if matches[0].start() > 0:
        lead_raw = _unmask_code_fences(masked[: matches[0].start()], fence_blocks)
        lead = _clean_markdown_text(lead_raw)

        if lead:
            add_markdown_blocks(builder, split_markdown_tables(lead))

    for i, m in enumerate(matches):
        builder.heading(len(m.group(1)), m.group(2).strip())
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(masked)
        content_raw = _unmask_code_fences(masked[start:end], fence_blocks)
        content = _clean_markdown_text(content_raw)

        if content:
            add_markdown_blocks(builder, split_markdown_tables(content))

    return builder.build()


def add_markdown_blocks(builder: DocumentTreeBuilder, segments: list[tuple[str, str]]) -> None:
    for ctype, text in segments:
        builder.block(text, NodeKind.TABLE if ctype == PageContentType.TABLE.value else NodeKind.PARAGRAPH)


def parse_markdown_sections(file_path: Path) -> list[Section]:
    """Preview DTOs of the markdown tree blocks, with table context attached."""
    return table_context_sections(sections_from_tree(parse_markdown_tree(file_path)))


def parse_markdown(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    text = _normalize_setext_headers(text)
    return _clean_markdown_text(text)


def extract_date_from_filename(filename: str) -> str | None:
    """Find YYYY-MM-DD or DD.MM.YYYY date in filename — simple heuristic for metadata."""
    m = _DATE_IN_FILENAME_RE.search(filename)

    if not m:
        return None

    date_str = m.group(1)

    # Normalize DD.MM.YYYY to YYYY-MM-DD
    if "." in date_str:
        parts = date_str.split(".")
        date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"

    return date_str


def extract_doc_title(file_path: Path) -> str | None:
    """Return the document's first top-level heading as its title, if any.

    Used as a document-level "doc_title" metadata value so every chunk from
    this file stays attributable to it even without the rest of the
    document for context (see langchain_document_parser.py). Falls back to
    the filename at the caller if this returns None.
    """
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    raw = _normalize_setext_headers(raw)
    masked, _blocks = _mask_code_fences(raw)
    m = _MD_HEADER_RE.search(masked)

    if not m:
        return None

    return m.group(2).strip() or None
