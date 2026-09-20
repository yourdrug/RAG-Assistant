"""RTF walker — tokenizer and structural parser for RTF documents.

Converts raw RTF into a list of ``Segment`` objects (paragraphs and tables),
preserving document order and formatting metadata (bold, font size, centering).

Extracted from ``rtf.py`` to reduce its size and isolate the walker concern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


# --- Constants --------------------------------------------------------------

_SKIP_DESTINATIONS = {
    "fonttbl",
    "colortbl",
    "stylesheet",
    "info",
    "generator",
    "pict",
    "object",
    "objdata",
    "themedata",
    "colorschememapping",
    "latentstyles",
    "rsidtbl",
    "listtable",
    "listoverridetable",
    "revtbl",
    "xmlnstbl",
    "footnote",
    "atnid",
    "atnauthor",
    "atndate",
}

_CONTROL_WORD_RE = re.compile(r"\\([a-zA-Z]+)(-?\d+)?[ ]?")
_HEX_ESCAPE_RE = re.compile(r"\\'([0-9a-fA-F]{2})")


# --- Dataclasses -----------------------------------------------------------


@dataclass
class Paragraph:
    text: str
    font_size: float | None
    bold: bool
    centered: bool


@dataclass
class Table:
    rows: list[list[str]]


@dataclass
class Segment:
    paragraph: Paragraph | None = None
    table: Table | None = None


class GroupState:
    __slots__ = ("bold", "font_size", "centered", "skip")

    def __init__(self, bold=False, font_size=None, centered=False, skip=False):
        self.bold = bold
        self.font_size = font_size
        self.centered = centered
        self.skip = skip

    def copy(self) -> "GroupState":
        return GroupState(self.bold, self.font_size, self.centered, self.skip)


@dataclass
class WalkerState:
    """Mutable state shared across helper functions during _walk_paragraphs."""

    stack: list[GroupState] = field(default_factory=lambda: [GroupState()])
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
    segments: list[Segment] = field(default_factory=list)
    codepage: str = "cp1251"


# --- Walker helpers --------------------------------------------------------


def _flush_hex(ws: WalkerState) -> None:
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


def _flush_cell(ws: WalkerState) -> None:
    """Finalize current table cell and append to current row."""
    if ws.current_cell_chars:
        cell_text = re.sub(r"[ \t]+", " ", "".join(ws.current_cell_chars)).strip()
        ws.current_row.append(cell_text)
        ws.current_cell_chars = []


def _flush_paragraph(ws: WalkerState) -> None:
    """Finalize current paragraph text and append as a Segment."""
    _flush_hex(ws)
    text = re.sub(r"[ \t]+", " ", "".join(ws.cur_chars)).strip()
    if text:
        ws.segments.append(
            Segment(
                paragraph=Paragraph(
                    text=text,
                    font_size=ws.cur_max_size,
                    bold=ws.cur_all_bold and ws.cur_any_text,
                    centered=ws.cur_centered,
                )
            )
        )
    ws.cur_chars, ws.cur_any_text, ws.cur_all_bold = [], False, True
    ws.cur_max_size, ws.cur_centered = None, False


def _flush_table(ws: WalkerState) -> None:
    """Finalize current table and append as a Segment."""
    _flush_cell(ws)
    if ws.current_row:
        ws.table_rows.append(ws.current_row)
        ws.current_row = []
    if ws.table_rows:
        ws.segments.append(Segment(table=Table(rows=ws.table_rows)))
        ws.table_rows = []
    ws.in_table = False


def _get_target(ws: WalkerState) -> list[str]:
    """Return the active character buffer (table cell or paragraph)."""
    return ws.current_cell_chars if ws.in_table else ws.cur_chars


def _handle_pard(ws: WalkerState, top: GroupState, argval: int | None) -> None:
    top.bold, top.font_size, top.centered = False, None, False


def _handle_skip_dest(ws: WalkerState, top: GroupState, argval: int | None) -> None:
    top.skip = True


def _handle_bold(ws: WalkerState, top: GroupState, argval: int | None) -> None:
    top.bold = argval != 0 if argval is not None else True


def _handle_font_size(ws: WalkerState, top: GroupState, argval: int | None) -> None:
    if argval is not None:
        top.font_size = argval / 2.0


def _handle_tab(ws: WalkerState, top: GroupState, argval: int | None) -> None:
    if not top.skip:
        _get_target(ws).append("\t")


def _handle_dash(ws: WalkerState, top: GroupState, argval: int | None) -> None:
    if not top.skip:
        _get_target(ws).append("-")


def _handle_intbl(ws: WalkerState, top: GroupState, argval: int | None) -> None:
    if not ws.in_table:
        ws.in_table = True


def _handle_cell(ws: WalkerState, top: GroupState, argval: int | None) -> None:
    if ws.in_table:
        _flush_cell(ws)


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


def _update_cur_metrics(ws: WalkerState, top: GroupState) -> None:
    """Propagate font-size and centering from the current group state."""
    if not top.skip:
        if top.font_size is not None:
            ws.cur_max_size = (
                top.font_size if ws.cur_max_size is None else max(ws.cur_max_size, top.font_size)
            )
        if top.centered:
            ws.cur_centered = True


def _handle_row(ws: WalkerState, m: re.Match, rtf: str, n: int) -> None:
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


def _process_control_word(
    ws: WalkerState, word: str, argval: int | None, m: re.Match, rtf: str, n: int
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


def _process_hex_escape(ws: WalkerState, rtf: str, pos: int) -> int:
    r"""Process \\'xx hex escape. Returns new position."""
    hexm = _HEX_ESCAPE_RE.match(rtf, pos)
    if hexm:
        if not ws.stack[-1].skip:
            ws.hex_buffer.append(int(hexm.group(1), 16))
            if not ws.stack[-1].bold:
                ws.hex_run_bold = False
        return hexm.end()
    return pos + 1


def _process_special_symbol(ws: WalkerState, nxt: str) -> None:
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


def _process_backslash(ws: WalkerState, rtf: str, pos: int, n: int) -> int:
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


def _finalize_walker(ws: WalkerState) -> list[Segment]:
    """Flush any remaining content and return the segment list."""
    if ws.in_table:
        _flush_cell(ws)
        if ws.current_row:
            ws.table_rows.append(ws.current_row)
        if ws.table_rows:
            ws.segments.append(Segment(table=Table(rows=ws.table_rows)))
    else:
        _flush_paragraph(ws)
    return ws.segments


# --- Main walker -----------------------------------------------------------


def _walk_paragraphs(rtf: str, codepage: str = "cp1251") -> list[Segment]:
    r"""Tokenize RTF into paragraphs and tables, preserving document order."""
    ws = WalkerState(codepage=codepage)
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
