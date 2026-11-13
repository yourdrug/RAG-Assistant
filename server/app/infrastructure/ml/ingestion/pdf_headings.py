"""Recover PDF section paths while retaining text and physical page locations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
import re
from typing import TYPE_CHECKING, Any

import fitz
from langchain.schema import Document

from domain.ingestion.boundary_validation import BoundaryCandidate, validate_candidates
from domain.ingestion.profiles.types import BoundaryLevel
from domain.knowledge.enums import PageContentType
from infrastructure.ml.ingestion.utils import clean_pdf_text


if TYPE_CHECKING:
    from collections.abc import Sequence


MIN_HEADINGS = 2
MAX_HEADING_CHARS = 120
# A body-font numbered line this close to the longest line on the page is wrapped prose.
MIN_WRAPPED_LINE_CHARS = 60
WRAPPED_LINE_SHARE = 0.8
OCR_HEADING_KINDS = frozenset({"title", "section-header", "section_header"})
NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?[ \t]+[А-ЯЁA-Z]")
NUMBER_LEVELS = [
    BoundaryLevel("point", NUMBERED_HEADING_RE),
    BoundaryLevel("subpoint_num", NUMBERED_HEADING_RE),
]


@dataclass(frozen=True)
class PDFHeading:
    document: int
    start: int
    level: int
    title: str


def pdf_line_styles(doc: fitz.Document) -> tuple[dict[int, list[dict[str, Any]]], float]:
    """Character-weighted font mode avoids treating a repeated short title as body."""
    styles: dict[int, list[dict[str, Any]]] = {}
    sizes: Counter[float] = Counter()
    flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES

    for index in range(len(doc)):
        lines = []

        for block in doc.load_page(index).get_text("dict", flags=flags)["blocks"]:
            for line in block.get("lines", []):
                spans = [span for span in line["spans"] if span["text"].strip()]

                if not spans:
                    continue

                for span in spans:
                    sizes[round(span["size"], 1)] += len(span["text"].strip())

                lines.append(
                    {
                        "text": clean_pdf_text("".join(span["text"] for span in spans)),
                        "size": max(span["size"] for span in spans),
                        "bold": all(span["flags"] & 16 for span in spans),
                        "indent": line["bbox"][0],
                    }
                )

        styles[index + 1] = lines

    return styles, sizes.most_common(1)[0][0] if sizes else 0.0


def line_hints(item: Document, styles: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve styles against surviving cleaned text; table and boilerplate text stay out."""
    by_text: dict[str, list[dict[str, Any]]] = {}

    for style in styles:
        by_text.setdefault(style["text"], []).append(style)

    hints = []

    for match in re.finditer(r"[^\n]+", item.page_content):
        text = match.group().strip()
        available = by_text.get(text, [])
        style = available.pop(0) if available else {}
        hints.append({**style, "text": text, "start": match.start(), "end": match.end()})

    return hints


def inferred_headings(pages: list[Document], base_size: float) -> list[PDFHeading]:
    font_candidates: list[tuple[int, dict[str, Any]]] = []
    numbered: list[BoundaryCandidate] = []
    locations: dict[int, tuple[int, int, str]] = {}
    offset = 0
    text = "\n\n".join(item.page_content for item in pages)

    for index, item in enumerate(pages):
        hints = item.metadata.get("line_hints", [])

        for position, hint in enumerate(hints):
            title = hint["text"]
            short = bool(
                title
                and len(title) < MAX_HEADING_CHARS
                and not title.endswith((".", ":", ";", ",", "!", "?"))
            )
            font_heading = short and bool(
                (base_size and hint.get("size", 0) >= base_size * 1.15) or hint.get("bold")
            )

            match = NUMBERED_HEADING_RE.match(title) if short else None

            # Without a font signal, a numbered line is a heading only when it is not
            # the first line of a wrapped clause: "1.1. Стороны договорились о" followed
            # by "порядке ...", or a line that fills the text width.
            if match and not font_heading and (continues_line(hints, position) or fills_width(hints, title)):
                match = None

            if font_heading and not match:
                font_candidates.append((index, hint))

            if match:
                start = offset + hint["start"]
                kind = "subpoint_num" if "." in match.group(1) else "point"
                before = text[:start].rstrip()
                numbered.append(
                    BoundaryCandidate(
                        kind,
                        match.group(1),
                        start,
                        start + match.end(),
                        line_start=True,
                        prev_char=before[-1:],
                        next_text=title[match.end() :],
                        format_confirmed=font_heading,
                    )
                )
                locations[start] = (index, hint["start"], title)

        offset += len(item.page_content) + 2

    size_levels = heading_size_levels(font_candidates, base_size)
    result = {
        (index, hint["start"]): PDFHeading(
            index,
            hint["start"],
            size_levels.get(round(hint.get("size", 0), 1), max(size_levels.values(), default=0) + 1),
            hint["text"],
        )
        for index, hint in font_candidates
    }

    for candidate in validate_candidates(numbered, text, NUMBER_LEVELS):
        index, start, title = locations[candidate.start]
        result[index, start] = PDFHeading(index, start, len((candidate.value or "").split(".")), title)

    return sorted(result.values(), key=lambda heading: (heading.document, heading.start))


def continues_line(hints: Sequence[dict[str, Any]], position: int) -> bool:
    following = hints[position + 1]["text"] if position + 1 < len(hints) else ""
    return following[:1].islower()


def fills_width(hints: Sequence[dict[str, Any]], title: str) -> bool:
    longest = max((len(hint["text"]) for hint in hints), default=0)
    return longest >= MIN_WRAPPED_LINE_CHARS and len(title) >= longest * WRAPPED_LINE_SHARE


def heading_size_levels(candidates: list[tuple[int, dict[str, Any]]], base_size: float) -> dict[float, int]:
    sizes = sorted(
        {
            round(hint.get("size", 0), 1)
            for _, hint in candidates
            if hint.get("size", 0) > 0 and hint.get("size", 0) >= base_size * 1.15
        },
        reverse=True,
    )
    levels = {}
    cluster = 0.0
    level = 0

    for size in sizes:
        if not cluster or cluster >= size * 1.10:
            cluster = size
            level += 1

        levels[size] = level

    return levels


def toc_headings(pages: list[Document], toc: list[list[Any]]) -> list[PDFHeading]:
    result = []
    used: set[tuple[int, int]] = set()

    for level, raw_title, page, *_ in toc:
        title = clean_pdf_text(raw_title).strip()

        # An empty pattern would match any blank line and inject "" into the path.
        if not title:
            continue

        for index, item in enumerate(pages):
            if (
                item.metadata.get("page") != page
                or item.metadata.get("content_type") == PageContentType.TABLE.value
            ):
                continue

            match = re.search(rf"(?m)^[ \t]*{re.escape(title)}[ \t]*$", item.page_content)

            if match and (index, match.start()) not in used:
                result.append(PDFHeading(index, match.start(), level, title))
                used.add((index, match.start()))
                break

    return sorted(result, key=lambda heading: (heading.document, heading.start))


def attach_pdf_line_hints(pages: list[Document], styles: dict[int, list[dict[str, Any]]]) -> None:
    for item in pages:
        if item.metadata.get("content_type") == PageContentType.TABLE.value:
            continue

        # OCR text must never borrow font signals from a discarded text layer.
        hints = line_hints(
            item, [] if item.metadata.get("text_quality") == 0 else styles.get(item.metadata["page"], [])
        )

        if item.metadata.get("content_type") in OCR_HEADING_KINDS:
            for hint in hints:
                hint["bold"] = True

            # The OCR layout label is a heading signal only; once the block merges
            # with its body, the chunk must keep the ordinary text content type.
            item.metadata["content_type"] = PageContentType.TEXT.value

        item.metadata["line_hints"] = hints


def apply_pdf_headings(doc: fitz.Document, pages: list[Document]) -> list[Document]:
    styles, base_size = pdf_line_styles(doc)
    attach_pdf_line_hints(pages, styles)
    toc = doc.get_toc()
    headings = toc_headings(pages, toc) if toc else inferred_headings(pages, base_size)

    if len(headings) < MIN_HEADINGS:
        return pages

    sections = section_pdf_documents(pages, headings)

    if not toc:
        for section in sections:
            section.metadata["inferred_heading"] = True

    return sections


def section_pdf_documents(pages: list[Document], headings: list[PDFHeading]) -> list[Document]:
    by_document: dict[int, list[PDFHeading]] = {}

    for heading in headings:
        by_document.setdefault(heading.document, []).append(heading)

    stack: list[PDFHeading] = []
    result = []

    for index, item in enumerate(pages):
        current = by_document.get(index, [])
        starts = {heading.start: heading for heading in current}
        cuts = sorted({0, len(item.page_content), *starts})

        for low, high in pairwise(cuts):
            if current_heading := starts.get(low):
                while stack and stack[-1].level >= current_heading.level:
                    stack.pop()

                stack.append(current_heading)

            section = pdf_section_slice(item, low, high, frozenset(starts))

            if section is not None:
                if stack:
                    section.metadata["section"] = " > ".join(heading.title for heading in stack)

                result.append(section)

    return result


def pdf_section_slice(item: Document, low: int, high: int, heading_starts: frozenset[int]) -> Document | None:
    part = item.page_content[low:high]
    content = part.strip()

    if not content:
        return None

    start = low + len(part) - len(part.lstrip())
    metadata = dict(item.metadata)
    metadata["line_hints"] = [
        {
            **hint,
            "start": hint["start"] - start,
            "end": min(hint["end"], high) - start,
            "heading": hint["start"] in heading_starts,
        }
        for hint in item.metadata.get("line_hints", [])
        if start <= hint["start"] < high
    ]
    return Document(page_content=content, metadata=metadata)
