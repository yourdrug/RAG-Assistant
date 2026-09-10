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
    return rtf_to_text(_read_rtf_text(file_path)), {}


_TITLE_GROUP_RE = re.compile(r"\{\\title([^{}\\]*)\}")


def extract_doc_title(file_path: Path) -> str | None:
    """Best-effort extraction of the \\info\\title field, if the RTF has one.

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
    """Read the document's declared \\ansicpg codepage for \\'xx byte decoding."""
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


class _GroupState:
    __slots__ = ("bold", "font_size", "centered", "skip")

    def __init__(self, bold=False, font_size=None, centered=False, skip=False):
        self.bold = bold
        self.font_size = font_size
        self.centered = centered
        self.skip = skip

    def copy(self) -> "_GroupState":
        return _GroupState(self.bold, self.font_size, self.centered, self.skip)


def _walk_paragraphs(rtf: str) -> list[_Paragraph]:
    """Tokenize RTF into paragraphs, tracking bold/font-size/centering per paragraph."""
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

    paragraphs: list[_Paragraph] = []

    def flush_hex():
        nonlocal hex_buffer, cur_any_text, cur_all_bold, hex_run_bold
        if hex_buffer:
            if not stack[-1].skip:
                cur_chars.append(bytes(hex_buffer).decode(codepage, errors="replace"))
                cur_any_text = True
                if not hex_run_bold:
                    cur_all_bold = False
            hex_buffer = bytearray()
            hex_run_bold = True

    def flush_paragraph():
        nonlocal cur_chars, cur_any_text, cur_all_bold, cur_max_size, cur_centered
        flush_hex()
        text = re.sub(r"[ \t]+", " ", "".join(cur_chars)).strip()
        if text:
            paragraphs.append(
                _Paragraph(text=text, font_size=cur_max_size, bold=cur_all_bold and cur_any_text, centered=cur_centered)
            )
        cur_chars, cur_any_text, cur_all_bold, cur_max_size, cur_centered = [], False, True, None, False

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
                        cur_chars.append("\t")
                elif word in ("emdash", "endash"):
                    if not top.skip:
                        cur_chars.append("-")

                if not top.skip:
                    if top.font_size is not None:
                        cur_max_size = top.font_size if cur_max_size is None else max(cur_max_size, top.font_size)
                    if top.centered:
                        cur_centered = True
                pos = m.end()
                continue

            # Non-letter control symbol: \*, \~, \-, \_, \\, \{, \} etc.
            if nxt in ("\\", "{", "}"):
                if not top.skip:
                    cur_chars.append(nxt)
                    cur_any_text = True
            elif nxt == "~":
                if not top.skip:
                    cur_chars.append("\u00a0")
            elif nxt == "_":
                if not top.skip:
                    cur_chars.append("-")
            # anything else (\*, \-, unknown symbols): consumed, no output
            pos += 2
            continue

        # Plain character
        if not top.skip:
            flush_hex()
            cur_chars.append(ch)
            if ch.strip():
                cur_any_text = True
                if not top.bold:
                    cur_all_bold = False
        pos += 1

    flush_paragraph()
    return paragraphs


def _most_common_size(paragraphs: list[_Paragraph]) -> float | None:
    sizes = [p.font_size for p in paragraphs if p.font_size]
    if not sizes:
        return None
    return Counter(sizes).most_common(1)[0][0]


def parse_rtf_sections(file_path: Path) -> list[tuple[str | None, str]]:
    r"""Split RTF into (heading, content) sections using paragraph formatting.

    A paragraph is treated as a heading if it's short (<= 12 words) and
    either matches a common numbered-heading pattern ("Глава 1", "Статья 5",
    "Section 3", ...) or is bold in a font size larger than the document's
    baseline (most common) body size. Distinct heading sizes are ranked
    (largest = level 1) to build a breadcrumb path the same way
    parse_docx_sections/parse_markdown_sections do.

    This is a heuristic, not a guarantee: some RTF exports flatten all
    formatting to a single style, in which case no headings will be
    detected. Callers should treat a single (None, text) result as "nothing
    to split further" and are free to fall back to the flat parse_rtf output
    in that case.
    """
    raw = _read_rtf_text(file_path)
    paragraphs = _walk_paragraphs(raw)
    if not paragraphs:
        return [(None, rtf_to_text(raw))]

    baseline = _most_common_size(paragraphs)

    def is_heading(p: _Paragraph) -> bool:
        word_count = len(p.text.split())
        if word_count == 0 or word_count > _MAX_HEADING_WORDS:
            return False
        if _NUMBERED_HEADING_RE.match(p.text):
            return True
        return bool(baseline is not None and p.bold and p.font_size and p.font_size > baseline)

    heading_sizes = sorted(
        {p.font_size for p in paragraphs if p.font_size and is_heading(p)},
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

    for p in paragraphs:
        if is_heading(p):
            flush()
            level = size_to_level.get(p.font_size, fallback_level) if p.font_size else fallback_level
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, p.text))
            current_heading = " > ".join(h for _, h in stack)
            current_lines = []
        else:
            current_lines.append(p.text)
    flush()

    if not sections:
        return [(None, "\n".join(p.text for p in paragraphs))]
    return sections
