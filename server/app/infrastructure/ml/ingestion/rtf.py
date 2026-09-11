"""RTF parsing — striprtf wrapper + heuristic structural section splitting."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from striprtf.striprtf import rtf_to_text


# --- Raw byte -> text decoding ---------------------------------------------


def _read_rtf_text(file_path: Path) -> str:
    """Read raw RTF bytes, trying encodings that preserve the RTF structure."""
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
    info = _extract_rtf_info(raw)
    meta.update(info)
    meta["codepage"] = _detect_codepage(raw)
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
    r"""Best-effort extraction of the \\info\\title field."""
    raw = _read_rtf_text(file_path)
    m = _TITLE_GROUP_RE.search(raw[:4000])
    if not m:
        return None
    title = m.group(1).strip()
    return title or None


# --- Constants & dataclasses -----------------------------------------------

_SKIP_DESTINATIONS = {
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

_NUMBERED_HEADING_RE = re.compile(
    r"^(Глава|Раздел|Часть|Статья|§|Пункт|Chapter|Section|Article|Appendix|Приложение)\s+\S",
    re.IGNORECASE,
)
_MAX_HEADING_WORDS = 12


def _detect_codepage(rtf: str) -> str:
    m = _ANSICPG_RE.search(rtf[:1000])
    if m:
        return _CODEPAGE_MAP.get(m.group(1), "cp1251")
    return "cp1251"


@dataclass
class _Paragraph:
    text: str
    font_size: float | None
    bold: bool
    centered: bool


@dataclass
class _Table:
    rows: list[list[str]]


@dataclass
class _Segment:
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


@dataclass
class _WalkerState:
    """Mutable state shared across helper functions during _walk_paragraphs."""

    stack: list[_GroupState] = field(default_factory=lambda: [_GroupState()])
    cur_chars: list[str] = field(default_factory=list)
    cur_any_text: bool = False
    cur_all_bold: bool = True
    cur_max_size: float | None = None
    cur_centered: bool = False
    hex_buffer: bytearray = field(default_factory=bytearray)
    hex_run_bold: bool = True
    in_table: bool = False
    table_rows: list[list[str]] = field(default_factory=list)
    current_row: list[str] = field(default_factory=list)
    current_cell_chars: list[str] = field(default_factory=list)
    segments: list[_Segment] = field(default_factory=list)
    codepage: str = "cp1251"


# --- Walker helpers (extracted from _walk_paragraphs) -----------------------


def _flush_hex(ws: _WalkerState) -> None:
    """Decode accumulated hex bytes and append to current buffer."""
    if ws.hex_buffer:
        if not ws.stack[-1].skip:
            text = bytes(ws.hex_buffer).decode(ws.codepage, errors="replace")
            if ws.in_table:
                ws.current_cell_chars.append(text)
            else:
                ws.cur_chars.append(text)
            ws.cur_any_text = True
            if not ws.hex_run_bold:
                ws.cur_all_bold = False
        ws.hex_buffer = bytearray()
        ws.hex_run_bold = True


def _flush_cell(ws: _WalkerState) -> None:
    """Finalize current table cell and append to current row."""
    if ws.current_cell_chars:
        cell_text = re.sub(r"[ \t]+", " ", "".join(ws.current_cell_chars)).strip()
        ws.current_row.append(cell_text)
        ws.current_cell_chars = []


def _flush_paragraph(ws: _WalkerState) -> None:
    """Finalize current paragraph text and append as a _Segment."""
    _flush_hex(ws)
    text = re.sub(r"[ \t]+", " ", "".join(ws.cur_chars)).strip()
    if text:
        ws.segments.append(
            _Segment(
                paragraph=_Paragraph(
                    text=text,
                    font_size=ws.cur_max_size,
                    bold=ws.cur_all_bold and ws.cur_any_text,
                    centered=ws.cur_centered,
                )
            )
        )
    ws.cur_chars, ws.cur_any_text, ws.cur_all_bold = [], False, True
    ws.cur_max_size, ws.cur_centered = None, False


def _flush_table(ws: _WalkerState) -> None:
    """Finalize current table and append as a _Segment."""
    _flush_cell(ws)
    if ws.current_row:
        ws.table_rows.append(ws.current_row)
        ws.current_row = []
    if ws.table_rows:
        ws.segments.append(_Segment(table=_Table(rows=ws.table_rows)))
        ws.table_rows = []
    ws.in_table = False


def _get_target(ws: _WalkerState) -> list[str]:
    """Return the active character buffer (table cell or paragraph)."""
    return ws.current_cell_chars if ws.in_table else ws.cur_chars


def _handle_pard(ws: _WalkerState, top: _GroupState, argval: int | None) -> None:
    top.bold, top.font_size, top.centered = False, None, False


def _handle_skip_dest(ws: _WalkerState, top: _GroupState, argval: int | None) -> None:
    top.skip = True


def _handle_bold(ws: _WalkerState, top: _GroupState, argval: int | None) -> None:
    top.bold = argval != 0 if argval is not None else True


def _handle_font_size(ws: _WalkerState, top: _GroupState, argval: int | None) -> None:
    if argval is not None:
        top.font_size = argval / 2.0


def _handle_tab(ws: _WalkerState, top: _GroupState, argval: int | None) -> None:
    if not top.skip:
        _get_target(ws).append("\t")


def _handle_dash(ws: _WalkerState, top: _GroupState, argval: int | None) -> None:
    if not top.skip:
        _get_target(ws).append("-")


def _handle_intbl(ws: _WalkerState, top: _GroupState, argval: int | None) -> None:
    if not ws.in_table:
        ws.in_table = True


def _handle_cell(ws: _WalkerState, top: _GroupState, argval: int | None) -> None:
    if ws.in_table:
        _flush_cell(ws)


# Dispatch table: maps control-word strings to handler(ws, top, argval) callables.
_CENTERING_WORDS = frozenset({"qc", "ql", "qr", "qj"})

_CONTROL_HANDLERS: dict[str, Callable[..., None]] = {
    "pard": _handle_pard,
    "b": _handle_bold,
    "fs": _handle_font_size,
    "tab": _handle_tab,
    "emdash": _handle_dash,
    "endash": _handle_dash,
    "intbl": _handle_intbl,
    "cell": _handle_cell,
}


def _update_cur_metrics(ws: _WalkerState, top: _GroupState) -> None:
    """Propagate font-size and centering from the current group state."""
    if not top.skip:
        if top.font_size is not None:
            ws.cur_max_size = (
                top.font_size if ws.cur_max_size is None
                else max(ws.cur_max_size, top.font_size)
            )
        if top.centered:
            ws.cur_centered = True


def _process_control_word(
    ws: _WalkerState, word: str, argval: int | None, m: re.Match, rtf: str, n: int
) -> bool:
    """Process an RTF control word. Returns True if word was handled."""
    top = ws.stack[-1]

    if word in ("par", "sect", "page"):
        if ws.in_table:
            _flush_cell(ws)
        else:
            _flush_paragraph(ws)
        return True

    if word in _SKIP_DESTINATIONS:
        _handle_skip_dest(ws, top, argval)
        _update_cur_metrics(ws, top)
        return True

    if word == "row":
        _handle_row(ws, m, rtf, n)
        _update_cur_metrics(ws, top)
        return True

    if word in _CENTERING_WORDS:
        top.centered = word == "qc"
        _update_cur_metrics(ws, top)
        return True

    handler = _CONTROL_HANDLERS.get(word)
    if handler:
        handler(ws, top, argval)
    else:
        return False

    _update_cur_metrics(ws, top)
    return True


def _handle_row(ws: _WalkerState, m: re.Match, rtf: str, n: int) -> None:
    r"""Handle \\row control word with peek-ahead for table finalization."""
    if not ws.in_table:
        return
    _flush_cell(ws)
    if ws.current_row:
        ws.table_rows.append(ws.current_row)
        ws.current_row = []
    if ws.table_rows and not ws.current_row:
        peek_pos = m.end()
        while peek_pos < n and rtf[peek_pos] in (" ", "\n", "\r"):
            peek_pos += 1
        next_5 = rtf[peek_pos : peek_pos + 5] if peek_pos < n else ""
        if next_5 not in ("\\cell", "\\intb", "\\row"):
            _flush_table(ws)


def _process_hex_escape(ws: _WalkerState, rtf: str, pos: int) -> int:
    r"""Process \\'xx hex escape. Returns new position."""
    hexm = _HEX_ESCAPE_RE.match(rtf, pos)
    if hexm:
        if not ws.stack[-1].skip:
            ws.hex_buffer.append(int(hexm.group(1), 16))
            if not ws.stack[-1].bold:
                ws.hex_run_bold = False
        return hexm.end()
    return pos + 1


def _process_special_symbol(ws: _WalkerState, nxt: str) -> None:
    r"""Process non-letter control symbols (\\\\, \\{, \\}, \\~, \\_)."""
    top = ws.stack[-1]
    if top.skip:
        return
    target = ws.current_cell_chars if ws.in_table else ws.cur_chars
    if nxt in ("\\", "{", "}"):
        target.append(nxt)
        ws.cur_any_text = True
    elif nxt == "~":
        target.append("\u00a0")
    elif nxt == "_":
        target.append("-")


def _process_backslash(ws: _WalkerState, rtf: str, pos: int, n: int) -> int:
    r"""Process a backslash escape at *pos*. Returns new position."""
    new_pos = _process_hex_escape(ws, rtf, pos)
    if new_pos != pos + 1:
        return new_pos

    _flush_hex(ws)
    nxt = rtf[pos + 1] if pos + 1 < n else ""
    if nxt.isalpha():
        m = _CONTROL_WORD_RE.match(rtf, pos)
        if not m:
            return pos + 1
        word, arg = m.group(1), m.group(2)
        argval = int(arg) if arg else None
        _process_control_word(ws, word, argval, m, rtf, n)
        return m.end()

    _process_special_symbol(ws, nxt)
    return pos + 2


def _finalize_walker(ws: _WalkerState) -> list[_Segment]:
    """Flush any remaining content and return the segment list."""
    if ws.in_table:
        _flush_cell(ws)
        if ws.current_row:
            ws.table_rows.append(ws.current_row)
        if ws.table_rows:
            ws.segments.append(_Segment(table=_Table(rows=ws.table_rows)))
    else:
        _flush_paragraph(ws)
    return ws.segments


# --- Main walker -----------------------------------------------------------


def _walk_paragraphs(rtf: str) -> list[_Segment]:
    r"""Tokenize RTF into paragraphs and tables, preserving document order."""
    ws = _WalkerState(codepage=_detect_codepage(rtf))
    pos, n = 0, len(rtf)

    while pos < n:
        ch = rtf[pos]
        top = ws.stack[-1]

        if ch == "{":
            _flush_hex(ws)
            ws.stack.append(top.copy())
            pos += 1
            continue
        if ch == "}":
            _flush_hex(ws)
            if len(ws.stack) > 1:
                ws.stack.pop()
            pos += 1
            continue
        if ch == "\\":
            pos = _process_backslash(ws, rtf, pos, n)
            continue
        # Plain character
        if not top.skip:
            _flush_hex(ws)
            target = ws.current_cell_chars if ws.in_table else ws.cur_chars
            target.append(ch)
            if ch.strip():
                ws.cur_any_text = True
                if not top.bold:
                    ws.cur_all_bold = False
        pos += 1

    return _finalize_walker(ws)


# --- Post-processing helpers -----------------------------------------------


def _most_common_size(segments: list[_Segment]) -> float | None:
    sizes = [s.paragraph.font_size for s in segments if s.paragraph and s.paragraph.font_size]
    if not sizes:
        return None
    return Counter(sizes).most_common(1)[0][0]


def _table_to_markdown(table: _Table) -> str:
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
    r"""Extract tables from RTF by detecting \\intbl/\\cell/\\row control words."""
    segments = _walk_paragraphs(rtf_raw)
    tables = []
    for seg in segments:
        if seg.table:
            md = _table_to_markdown(seg.table)
            if md:
                tables.append(md)
    return tables


def _is_heading(seg: _Segment, baseline: float | None) -> bool:
    """Return True if *seg* looks like a document heading."""
    p = seg.paragraph
    if p is None:
        return False
    word_count = len(p.text.split())
    if word_count == 0 or word_count > _MAX_HEADING_WORDS:
        return False
    if _NUMBERED_HEADING_RE.match(p.text):
        return True
    return bool(baseline is not None and p.bold and p.font_size and p.font_size > baseline)


def _build_heading_index(
    segments: list[_Segment], baseline: float | None
) -> tuple[dict[float, int], int]:
    """Map font sizes to heading levels; returns (size_to_level, fallback_level)."""
    heading_sizes = sorted(
        {
            s.paragraph.font_size
            for s in segments
            if s.paragraph and s.paragraph.font_size and _is_heading(s, baseline)
        },
        reverse=True,
    )
    size_to_level = {size: i + 1 for i, size in enumerate(heading_sizes)}
    fallback_level = len(heading_sizes) + 1
    return size_to_level, fallback_level


def _split_by_headings(
    segments: list[_Segment],
    baseline: float | None,
    size_to_level: dict[float, int],
    fallback_level: int,
) -> list[tuple[str | None, str]]:
    """Walk segments, splitting on detected headings into (heading, content) pairs."""
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
            md = _table_to_markdown(seg.table)
            if md:
                flush()
                sections.append((current_heading, "\x00TABLE:" + md))
                current_lines = []
        elif seg.paragraph:
            if _is_heading(seg, baseline):
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
    return sections


def _fallback_flat_text(segments: list[_Segment]) -> str:
    """Collect all paragraph text and table markdown into a single string."""
    all_text: list[str] = []
    for seg in segments:
        if seg.paragraph:
            all_text.append(seg.paragraph.text)
        elif seg.table:
            md = _table_to_markdown(seg.table)
            if md:
                all_text.append(md)
    return "\n".join(all_text)


def parse_rtf_sections(file_path: Path) -> list[tuple[str | None, str]]:
    r"""Split RTF into (heading, content) sections using paragraph formatting."""
    raw = _read_rtf_text(file_path)
    segments = _walk_paragraphs(raw)
    if not segments:
        return [(None, rtf_to_text(raw))]
    baseline = _most_common_size(segments)
    size_to_level, fallback_level = _build_heading_index(segments, baseline)
    sections = _split_by_headings(segments, baseline, size_to_level, fallback_level)
    if not sections:
        return [(None, _fallback_flat_text(segments))]
    return sections
