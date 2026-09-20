"""RTF parsing — striprtf wrapper + heuristic structural section splitting."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from striprtf.striprtf import rtf_to_text

# Re-export walker internals for backward compatibility (tests import these)
from infrastructure.ml.ingestion.rtf_walker import (  # noqa: F401
    Paragraph,
    Segment,
    Table,
    _walk_paragraphs,
)


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


_TITLE_GROUP_RE = re.compile(r"\{\\title([^{}\\]*)\}")
_AUTHOR_GROUP_RE = re.compile(r"\{\\author([^{}\\]*)\}")
_CREATED_GROUP_RE = re.compile(r"\{\\creatim[^}]*\\yr(\d+)[^}]*\\mo(\d+)[^}]*\\dy(\d+)[^}]*\}")
_PAGE_RE = re.compile(r"\\page\b")
_ANSICPG_RE = re.compile(r"\\ansicpg(\d+)")
_CODEPAGE_MAP = {
    "1250": "cp1250",
    "1251": "cp1251",
    "1252": "cp1252",
    "1253": "cp1253",
    "1254": "cp1254",
    "1257": "cp1257",
    "65001": "utf-8",
    "10000": "mac_roman",
}


def _detect_codepage(rtf: str) -> str:
    m = _ANSICPG_RE.search(rtf[:1000])
    if m:
        return _CODEPAGE_MAP.get(m.group(1), "cp1251")
    return "cp1251"


def _extract_rtf_info(rtf_raw: str) -> dict:
    r"""Extract author and creation date from the RTF \\info group."""
    result: dict = {}
    m = _TITLE_GROUP_RE.search(rtf_raw)
    if m:
        result["title"] = m.group(1).strip()
    m = _AUTHOR_GROUP_RE.search(rtf_raw)
    if m:
        result["author"] = m.group(1).strip()
    m = _CREATED_GROUP_RE.search(rtf_raw)
    if m:
        result["created_date"] = f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    return result


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


def extract_doc_title(file_path: Path) -> str | None:
    r"""Extract document title from RTF \\title group or metadata."""
    raw = _read_rtf_text(file_path)
    info = _extract_rtf_info(raw)
    return info.get("title")


# --- Post-processing helpers -----------------------------------------------

_NUMBERED_HEADING_RE = re.compile(
    r"^(Глава|Раздел|Часть|Статья|§|Пункт|Chapter|Section|Article|Appendix|Приложение)\s+\S",
    re.IGNORECASE,
)
_MAX_HEADING_WORDS = 12


def _most_common_size(segments: list[Segment]) -> float | None:
    sizes = [s.paragraph.font_size for s in segments if s.paragraph and s.paragraph.font_size]
    if not sizes:
        return None
    return Counter(sizes).most_common(1)[0][0]


def _table_to_markdown(table: Table) -> str:
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
    segments = _walk_paragraphs(rtf_raw, codepage=_detect_codepage(rtf_raw))
    tables = []
    for seg in segments:
        if seg.table:
            md = _table_to_markdown(seg.table)
            if md:
                tables.append(md)
    return tables


def _is_heading(seg: Segment, baseline: float | None) -> bool:
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


def _build_heading_index(segments: list[Segment], baseline: float | None) -> tuple[dict[float, int], int]:
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
    segments: list[Segment],
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


def _fallback_flat_text(segments: list[Segment]) -> str:
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
    segments = _walk_paragraphs(raw, codepage=_detect_codepage(raw))
    if not segments:
        return [(None, rtf_to_text(raw))]
    baseline = _most_common_size(segments)
    size_to_level, fallback_level = _build_heading_index(segments, baseline)
    sections = _split_by_headings(segments, baseline, size_to_level, fallback_level)
    if not sections:
        return [(None, _fallback_flat_text(segments))]
    return sections
