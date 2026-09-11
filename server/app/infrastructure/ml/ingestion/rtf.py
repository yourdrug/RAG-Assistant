"""RTF parsing — striprtf wrapper + heuristic structural section splitting."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from striprtf.striprtf import rtf_to_text


# --- Raw byte -> text decoding (unchanged) --------------------------------


def _read_rtf_text(file_path: Path) -> str:
    """Read raw RTF bytes, trying encodings that preserve the RTF structure.

    RTF files from older Russian systems are often saved as Windows-1251.
    Reading them as UTF-8 with errors="replace" corrupts RTF escape sequences
    and produces garbled output.  We try utf-8 first (most common for modern
    files), then fall back to cp1251, and finally latin-1 (never fails).
    """
    raw = file_path.read_bytes()
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return raw.decode("latin-1")


def parse_rtf(file_path: Path) -> tuple[str, dict]:
    """Parse RTF and return (text, metadata)."""
    raw = _read_rtf_text(file_path)
    rtf_text = rtf_to_text(raw)
    meta: dict = {}

    # Extract info fields from raw RTF
    info = _extract_rtf_info(raw)
    meta.update(info)

    # Save codepage
    meta["codepage"] = _detect_codepage(raw)

    # Estimate page count from \page markers
    page_count = len(_PAGE_RE.findall(raw))
    if page_count > 0:
        meta["estimated_page_count"] = page_count

    return rtf_text, meta


_TITLE_GROUP_RE = re.compile(r"\{\\title([^{}\\]*)\}")
_AUTHOR_GROUP_RE = re.compile(r"\{\\author([^{}\\]*)\}")
_CREATED_GROUP_RE = re.compile(r"\{\\creatim[^}]*\\yr(\d+)[^}]*\\mo(\d+)[^}]*\\dy(\d+)[^}]*\}")
_PAGE_RE = re.compile(r"\\page\b")


def _extract_rtf_info(rtf_raw: str) -> dict:
    r"""Extract author and creation date from the RTF \\info group."""
    result: dict = {}
    m = _AUTHOR_GROUP_RE.search(rtf_raw[:8000])
    if m:
        author = m.group(1).strip()
        if author:
            result["doc_author"] = author
    m = _CREATED_GROUP_RE.search(rtf_raw[:8000])
    if m:
        try:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                result["doc_created"] = f"{year:04d}-{month:02d}-{day:02d}"
        except (ValueError, TypeError):
            pass
    return result


def extract_doc_title(file_path: Path) -> str | None:
    r"""Best-effort extraction of the \\info\\title field, if the RTF has one.

    Word only writes this when a document title was explicitly set in the
    file's properties, so this is frequently absent — callers should fall
    back to the filename when this returns None.
    """
    raw = _read_rtf_text(file_path)
    m = _TITLE_GROUP_RE.search(raw[:4000])
    if not m:
        return None
    title = m.group(1).strip()
    return title or None


# --- Structural section splitting ------------------------------------------
#
# striprtf discards all formatting, so unlike DOCX ("Heading N" paragraph
# style) or Markdown ("#") there is no explicit, ready-made heading marker
# to key off for RTF. Instead we walk the raw RTF control-word stream
# ourselves, tracking just enough paragraph-level formatting (bold, font
# size, centering) to tell a heading-styled paragraph from body text, plus a
# handful of common numbered-heading text patterns. This is a small,
# purpose-built walker — not a full RTF parser — so it only tracks the state
# needed for that classification and otherwise falls back to
# unambiguous behavior when the input doesn't look confidently structured.

_SKIP_DESTINATIONS = {
    # Tables/metadata that hold no real document text: walking into these
    # groups would otherwise leak font names, color numbers, style
    # definitions, and embedded object data into the "paragraph" text.
    "fonttbl", "colortbl", "stylesheet", "info", "generator", "pict",
    "object", "objdata", "themedata", "colorschememapping", "latentstyles",
    "rsidtbl", "listtable", "listoverridetable", "revtbl", "xmlnstbl",
    "footnote", "atnid", "atnauthor", "atndate",
}

_CONTROL_WORD_RE = re.compile(r"\\([a-zA-Z]+)(-?\d+)?[ ]?")
_HEX_ESCAPE_RE = re.compile(r"\\'([0-9a-fA-F]{2})")
_ANSICPG_RE = re.compile(r"\\ansicpg(\d+)")
_CODEPAGE_MAP = {
    "1250": "cp1250", "1251": "cp1251", "1252": "cp1252", "1253": "cp1253",
    "1254": "cp1254", "1257": "cp1257", "65001": "utf-8", "10000": "mac_roman",
}
# Table detection control words
_TABLE_INTBL_RE = re.compile(r"\\intbl\b")
_TABLE_CELL_RE = re.compile(r"\\cell\b")
_TABLE_ROW_RE = re.compile(r"\\row\b")

# Common numbered/named heading patterns (Russian + English legal/business
# document vocabulary) — used as a fallback signal when a heading isn't
# distinguished purely by bold/font-size (some RTF exports flatten style
# formatting to plain body-sized text).
_NUMBERED_HEADING_RE = re.compile(
    r"^(Глава|Раздел|Часть|Статья|§|Пункт|Chapter|Section|Article|Appendix|Приложение)\s+\S",
    re.IGNORECASE,
)
_MAX_HEADING_WORDS = 12


def _detect_codepage(rtf: str) -> str:
    r"""Read the document's declared \\ansicpg codepage for \\'xx byte decoding."""
    m = _ANSICPG_RE.search(rtf[:1000])
    if m:
        return _CODEPAGE_MAP.get(m.group(1), "cp1251")
    return "cp1251"  # this pipeline's RTF inputs are predominantly Cyrillic


@dataclass
class _Paragraph:
    text: str
    font_size: float | None  # points; None if never set explicitly
    bold: bool  # True only if every character in the paragraph was bold
    centered: bool


@dataclass
class _Table:
    rows: list[list[str]]


@dataclass
class _Segment:
    """A paragraph or table from the RTF walker, preserving document order."""

    paragraph: _Paragraph | None = None
    table: _Table | None = None


class _GroupState:
    __slots__ = ("bold", "font_size", "centered", "skip")

    def __init__(self, bold=False, font_size=None, centered=False, skip=False):
        self.bold = bold
        self.font_size = font_size
        self.centered = centered
        self.skip = skip

    def copy(self) -> "_GroupState":
        return _GroupState(self.bold, self.font_size, self.centered, self.skip)


def _walk_paragraphs(rtf: str) -> list[_Segment]:  # noqa: C901
    r"""Tokenize RTF into paragraphs and tables, preserving document order.

    Table-aware: tracks \\intbl/\\cell/\\row control words to detect table
    boundaries. Returns a list of _Segment objects, each containing either
    a _Paragraph (text) or _Table (rows of cells).
    """
    codepage = _detect_codepage(rtf)
    pos, n = 0, len(rtf)
    stack = [_GroupState()]

    cur_chars: list[str] = []
    cur_any_text = False
    cur_all_bold = True
    cur_max_size: float | None = None
    cur_centered = False
    hex_buffer = bytearray()
    hex_run_bold = True

    segments: list[_Segment] = []

    # Table state
    in_table = False
    table_rows: list[list[str]] = []
    current_row: list[str] = []
    current_cell_chars: list[str] = []

    def flush_hex():
        nonlocal hex_buffer, cur_any_text, cur_all_bold, hex_run_bold
        if hex_buffer:
            if not stack[-1].skip:
                if in_table:
                    current_cell_chars.append(bytes(hex_buffer).decode(codepage, errors="replace"))
                else:
                    cur_chars.append(bytes(hex_buffer).decode(codepage, errors="replace"))
                cur_any_text = True
                if not hex_run_bold:
                    cur_all_bold = False
            hex_buffer = bytearray()
            hex_run_bold = True

    def flush_cell():
        nonlocal current_cell_chars
        if current_cell_chars:
            cell_text = re.sub(r"[ \t]+", " ", "".join(current_cell_chars)).strip()
            current_row.append(cell_text)
            current_cell_chars = []

    def flush_paragraph():
        nonlocal cur_chars, cur_any_text, cur_all_bold, cur_max_size, cur_centered
        flush_hex()
        text = re.sub(r"[ \t]+", " ", "".join(cur_chars)).strip()
        if text:
            segments.append(
                _Segment(
                    paragraph=_Paragraph(
                        text=text,
                        font_size=cur_max_size,
                        bold=cur_all_bold and cur_any_text,
                        centered=cur_centered,
                    )
                )
            )
        cur_chars, cur_any_text, cur_all_bold, cur_max_size, cur_centered = [], False, True, None, False

    def flush_table():
        nonlocal table_rows, current_row, in_table
        flush_cell()
        if current_row:
            table_rows.append(current_row)
            current_row = []
        if table_rows:
            segments.append(_Segment(table=_Table(rows=table_rows)))
            table_rows = []
        in_table = False

    while pos < n:
        ch = rtf[pos]
        top = stack[-1]

        if ch == "{":
            flush_hex()
            stack.append(top.copy())
            pos += 1
            continue
        if ch == "}":
            flush_hex()
            if len(stack) > 1:
                stack.pop()
            pos += 1
            continue

        if ch == "\\":
            hexm = _HEX_ESCAPE_RE.match(rtf, pos)
            if hexm:
                if not top.skip:
                    hex_buffer.append(int(hexm.group(1), 16))
                    if not top.bold:
                        hex_run_bold = False
                pos = hexm.end()
                continue

            flush_hex()
            nxt = rtf[pos + 1] if pos + 1 < n else ""
            if nxt.isalpha():
                m = _CONTROL_WORD_RE.match(rtf, pos)
                if not m:
                    pos += 1
                    continue
                word, arg = m.group(1), m.group(2)
                argval = int(arg) if arg else None

                if word in ("par", "sect", "page"):
                    if in_table:
                        # End of a paragraph inside a table row — flush cell
                        flush_cell()
                    else:
                        flush_paragraph()
                    pos = m.end()
                    continue
                if word == "pard":
                    top.bold, top.font_size, top.centered = False, None, False
                elif word in _SKIP_DESTINATIONS:
                    top.skip = True
                elif word == "b":
                    top.bold = argval != 0 if argval is not None else True
                elif word == "fs" and argval is not None:
                    top.font_size = argval / 2.0
                elif word == "qc":
                    top.centered = True
                elif word in ("ql", "qr", "qj"):
                    top.centered = False
                elif word == "tab":
                    if not top.skip:
                        if in_table:
                            current_cell_chars.append("\t")
                        else:
                            cur_chars.append("\t")
                elif word in ("emdash", "endash"):
                    if not top.skip:
                        if in_table:
                            current_cell_chars.append("-")
                        else:
                            cur_chars.append("-")
                # Table control words
                elif word == "intbl":
                    if not in_table:
                        in_table = True
                elif word == "cell":
                    if in_table:
                        flush_cell()
                elif word == "row":
                    if in_table:
                        flush_cell()
                        if current_row:
                            table_rows.append(current_row)
                            current_row = []
                        # A \row inside a nested group might end the table;
                        # check if we should finalize. We finalize on \row
                        # only when there are complete rows.
                        if table_rows and not current_row:
                            # Check if next token is also table-related
                            # by peeking ahead. If not, finalize.
                            peek_pos = m.end()
                            peek_skip = peek_pos
                            while peek_skip < n and rtf[peek_skip] in (" ", "\n", "\r"):
                                peek_skip += 1
                            if (
                                peek_skip < n
                                and rtf[peek_skip : peek_skip + 5] in ("\\cell", "\\intb", "\\row")
                            ):
                                pass  # still in table
                            else:
                                flush_table()

                if not top.skip:
                    if top.font_size is not None:
                        cur_max_size = (
                            top.font_size if cur_max_size is None
                            else max(cur_max_size, top.font_size)
                        )
                    if top.centered:
                        cur_centered = True
                pos = m.end()
                continue

            # Non-letter control symbol: \*, \~, \-, \_, \\, \{, \} etc.
            if nxt in ("\\", "{", "}"):
                if not top.skip:
                    if in_table:
                        current_cell_chars.append(nxt)
                    else:
                        cur_chars.append(nxt)
                    cur_any_text = True
            elif nxt == "~":
                if not top.skip:
                    if in_table:
                        current_cell_chars.append("\u00a0")
                    else:
                        cur_chars.append("\u00a0")
            elif nxt == "_":
                if not top.skip:
                    if in_table:
                        current_cell_chars.append("-")
                    else:
                        cur_chars.append("-")
            # anything else (\*, \-, unknown symbols): consumed, no output
            pos += 2
            continue

        # Plain character
        if not top.skip:
            flush_hex()
            if in_table:
                current_cell_chars.append(ch)
            else:
                cur_chars.append(ch)
            if ch.strip():
                cur_any_text = True
                if not top.bold:
                    cur_all_bold = False
        pos += 1

    # Flush remaining content
    if in_table:
        flush_cell()
        if current_row:
            table_rows.append(current_row)
            current_row = []
        if table_rows:
            segments.append(_Segment(table=_Table(rows=table_rows)))
    else:
        flush_paragraph()
    return segments


def _most_common_size(segments: list[_Segment]) -> float | None:
    sizes = [s.paragraph.font_size for s in segments if s.paragraph and s.paragraph.font_size]
    if not sizes:
        return None
    return Counter(sizes).most_common(1)[0][0]


def _table_to_markdown(table: _Table) -> str:
    """Convert a _Table to markdown table format."""
    rows = table.rows
    if not rows:
        return ""
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")
    header = "| " + " | ".join(rows[0]) + " |"
    separator = "|" + "|".join(["---"] * max_cols) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, separator] + body) if body else header


def extract_rtf_tables(rtf_raw: str) -> list[str]:
    r"""Extract tables from RTF by detecting \\intbl/\\cell/\\row control words.

    Returns a list of markdown-formatted table strings. This is a standalone
    utility for callers that need only the tables without the surrounding text.
    """
    segments = _walk_paragraphs(rtf_raw)
    tables = []
    for seg in segments:
        if seg.table:
            md = _table_to_markdown(seg.table)
            if md:
                tables.append(md)
    return tables


def parse_rtf_sections(file_path: Path) -> list[tuple[str | None, str]]:  # noqa: C901
    r"""Split RTF into (heading, content) sections using paragraph formatting.

    A paragraph is treated as a heading if it's short (<= 12 words) and
    either matches a common numbered-heading pattern ("Глава 1", "Статья 5",
    "Section 3", ...) or is bold in a font size larger than the document's
    baseline (most common) body size. Distinct heading sizes are ranked
    (largest = level 1) to build a breadcrumb path the same way
    parse_docx_sections/parse_markdown_sections do.

    Tables (\\intbl/\\cell/\\row) are detected and emitted as
    ``\\x00TABLE:``-prefixed sections so downstream code tags them with
    ``content_type = "table"`` for row-batched splitting.
    """
    raw = _read_rtf_text(file_path)
    segments = _walk_paragraphs(raw)
    if not segments:
        return [(None, rtf_to_text(raw))]

    baseline = _most_common_size(segments)

    def is_heading(seg: _Segment) -> bool:
        p = seg.paragraph
        if p is None:
            return False
        word_count = len(p.text.split())
        if word_count == 0 or word_count > _MAX_HEADING_WORDS:
            return False
        if _NUMBERED_HEADING_RE.match(p.text):
            return True
        return bool(baseline is not None and p.bold and p.font_size and p.font_size > baseline)

    heading_sizes = sorted(
        {s.paragraph.font_size for s in segments if s.paragraph and s.paragraph.font_size and is_heading(s)},
        reverse=True,
    )
    size_to_level = {size: i + 1 for i, size in enumerate(heading_sizes)}
    fallback_level = len(heading_sizes) + 1  # numbered-pattern headings without a distinct larger size

    sections: list[tuple[str | None, str]] = []
    stack: list[tuple[int, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush():
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_heading, content))

    for seg in segments:
        if seg.table:
            # Emit table as a \x00TABLE:-prefixed section
            md = _table_to_markdown(seg.table)
            if md:
                flush()
                sections.append((current_heading, "\x00TABLE:" + md))
                current_lines = []
        elif seg.paragraph:
            if is_heading(seg):
                flush()
                p = seg.paragraph
                level = size_to_level.get(p.font_size, fallback_level) if p.font_size else fallback_level
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, p.text))
                current_heading = " > ".join(h for _, h in stack)
                current_lines = []
            else:
                current_lines.append(seg.paragraph.text)
    flush()

    if not sections:
        all_text = []
        for seg in segments:
            if seg.paragraph:
                all_text.append(seg.paragraph.text)
            elif seg.table:
                md = _table_to_markdown(seg.table)
                if md:
                    all_text.append(md)
        return [(None, "\n".join(all_text))]
    return sections
