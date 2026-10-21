"""RTF parsing — striprtf wrapper + heuristic structural section splitting."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from collections import Counter
from pathlib import Path

from infrastructure.ml.ingestion.table_formatting import render_table
from infrastructure.ml.ingestion.sections import ParsedFormat, Section
from infrastructure.ml.ingestion.structural_metadata import unit_metadata
from infrastructure.ml.rtf_decree_parser import parse_decree_rtf

from striprtf.striprtf import rtf_to_text
from domain.knowledge.enums import PageContentType
from infrastructure.ml.ingestion.rtf_walker import Segment, Table, _walk_paragraphs

if TYPE_CHECKING:
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.decree import DecreeDomainProfile
    from domain.ingestion.profiles.registry import DomainProfileRegistry

log = logging.getLogger(__name__)
_DECREE_FINGERPRINT_PREFIX_CHARS = 3000
MIN_DOCUMENT_CHARS = 20


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
    return render_table(table.rows)


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
) -> list[Section]:
    """Walk segments, splitting on detected headings into (heading, content) pairs."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush():
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(Section(current_heading, content))

    for seg in segments:
        if seg.table:
            md = _table_to_markdown(seg.table)
            if md:
                flush()
                sections.append(Section(current_heading, md, PageContentType.TABLE))
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


def parse_rtf_sections(file_path: Path) -> list[Section]:
    r"""Split RTF into (heading, content) sections using paragraph formatting."""
    raw = _read_rtf_text(file_path)
    segments = _walk_paragraphs(raw, codepage=_detect_codepage(raw))
    if not segments:
        return [Section(None, rtf_to_text(raw))]
    baseline = _most_common_size(segments)
    size_to_level, fallback_level = _build_heading_index(segments, baseline)
    sections = _split_by_headings(segments, baseline, size_to_level, fallback_level)
    if not sections:
        return [Section(None, _fallback_flat_text(segments))]
    return sections


class RtfDocumentParser:
    """Apply decree and generic section fallbacks for ingestion."""

    def __init__(
        self,
        domain_registry: DomainProfileRegistry | None = None,
        domain_settings: DomainSettingsPort | None = None,
    ) -> None:
        self._domain_registry = domain_registry
        self._domain_settings = domain_settings

    def parse(self, file_path: Path) -> ParsedFormat:
        """RTF parsing: decree-structured split, generic heuristic split, or flat text.

        Tries the domain-specific decree splitter first (most precise when it
        applies). If that doesn't match, falls back to the generic
        formatting-based section splitter (parse_rtf_sections); if that
        finds no headings either (flat/unstructured RTF), falls back further
        to a single flat document — the same behavior as before this path
        existed, so plain RTF files never regress.
        """
        text, _meta = parse_rtf(file_path)
        if not text or len(text.strip()) < MIN_DOCUMENT_CHARS:
            message = "Too little text in document"
            raise RuntimeError(message)

        decree_docs = self.try_parse_decree_rtf(file_path, text)
        if decree_docs is not None:
            return decree_docs

        try:
            sections = parse_rtf_sections(file_path)
        except Exception:
            log.exception("Structured RTF parsing failed for %s — falling back to flat parse", file_path.name)
            sections = None

        page_meta = {}
        title = extract_doc_title(file_path)
        if title:
            page_meta["doc_title"] = title

        if sections and len(sections) > 1:
            return ParsedFormat(sections, page_meta)

        return ParsedFormat([Section(None, text)], page_meta, structured=False)

    def get_decree_profile(self) -> DecreeDomainProfile | None:
        if self._domain_registry is None or self._domain_settings is None:
            return None
        try:
            profile = self._domain_registry.get("decree")
        except KeyError:
            return None
        return profile if profile.content_boundaries() else None  # type: ignore[return-value]

    def try_parse_decree_rtf(self, file_path: Path, text: str) -> ParsedFormat | None:
        """Decree-structured RTF path; None = fall through to flat RTF parsing.

        Uses the same deterministic fingerprint as document classification,
        applied to the file prefix for speed (see TZ section 5.2).
        """
        profile = self.get_decree_profile()
        if profile is None or not profile.structural_fingerprint(text[:_DECREE_FINGERPRINT_PREFIX_CHARS]):
            return None
        try:
            if self._domain_settings is None:
                return None
            units, doc_metadata = parse_decree_rtf(file_path, profile, self._domain_settings)
        except Exception:
            log.exception("Decree RTF parsing failed for %s — falling back to flat parse", file_path.name)
            return None

        docs = []
        for unit in units:
            if not unit.content.strip():
                continue
            metadata = unit_metadata(unit, profile, {"source": file_path.name, **doc_metadata})
            docs.append(Section(None, unit.content, metadata=metadata))
        if not docs or sum(len(d.content) for d in docs) < MIN_DOCUMENT_CHARS:
            message = "Too little text in document"
            raise RuntimeError(message)
        log.info("Decree RTF detected: %d structural units from %s", len(docs), file_path.name)
        return ParsedFormat(docs, structured=False)
