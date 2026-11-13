"""DOCX parsing — python-docx based text/table/section extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING, Any

import docx
from docx.oxml import parse_xml
from docx.oxml.ns import qn

from domain.knowledge.enums import PageContentType
from infrastructure.ml.ingestion.docx_numbering import (
    ListNumberer,
    _get_numbering_formats,
    _paragraph_list_info,
    get_numbering_texts,
    xml_element,
)
from infrastructure.ml.ingestion.sections import Section


if TYPE_CHECKING:
    from pathlib import Path

    from docx.document import Document as WordDocument
    from docx.table import Table
    from docx.text.paragraph import Paragraph

from infrastructure.ml.ingestion.table_formatting import markdown_cell, render_table
from infrastructure.storage.docx_validation import validate_docx_archive


def _has_page_break(paragraph: Paragraph) -> bool:
    """Check if a paragraph contains a manual page break."""
    for run in paragraph.runs:
        for br in xml_element(run).findall(qn("w:br")):
            if br.get(qn("w:type")) == "page":
                return True

    return False


def _paragraph_full_text(paragraph: Paragraph) -> str:
    """Full paragraph text, including runs nested inside hyperlinks."""
    t_tag = qn("w:t")
    tab_tag = qn("w:tab")
    br_tag = qn("w:br")
    parts: list[str] = []

    for node in xml_element(paragraph).iter():
        if node.tag == t_tag:
            if node.text:
                parts.append(node.text)
        elif node.tag == tab_tag:
            parts.append("\t")
        elif node.tag == br_tag:
            parts.append("\n")

    return "".join(parts)


# --- Numbering format resolution — delegated to _docx_numbering ---


# --- Extraction helpers -----------------------------------------------------


def _extract_headers_footers(doc: WordDocument) -> dict[str, list[str]]:
    """Collect unique header/footer text across all sections."""
    headers: list[str] = []
    footers: list[str] = []
    seen_headers: set[str] = set()
    seen_footers: set[str] = set()

    for section in doc.sections:
        for container, seen, bucket in (
            (section.header, seen_headers, headers),
            (section.footer, seen_footers, footers),
        ):
            if container.is_linked_to_previous:
                continue

            text = "\n".join(p.text.strip() for p in container.paragraphs if p.text.strip())

            if text and text not in seen:
                seen.add(text)
                bucket.append(text)

    return {"headers": headers, "footers": footers}


def _extract_notes(doc: WordDocument, part_reltype_substring: str, xml_tag: str) -> list[str]:
    """Extract footnote or endnote body text."""
    notes_part = None

    for rel in doc.part.rels.values():
        if part_reltype_substring in rel.reltype:
            notes_part = rel.target_part
            break

    if notes_part is None:
        return []

    texts = []
    element = getattr(notes_part, "element", None)

    if element is None:
        element = parse_xml(notes_part.blob)

    for note in element.findall(qn(xml_tag)):
        note_id = note.get(qn("w:id"))

        if note_id in ("-1", "0"):
            continue

        note_text = "".join(t.text for t in note.iter(qn("w:t")) if t.text).strip()

        if note_text:
            texts.append(note_text)

    return texts


def _extract_document_properties(doc: WordDocument) -> dict[str, Any]:
    """Pull core document properties (title/author/dates) into metadata."""
    props = doc.core_properties
    meta: dict[str, Any] = {}

    if props.title:
        meta["doc_title"] = props.title

    if props.author:
        meta["doc_author"] = props.author

    if props.subject:
        meta["doc_subject"] = props.subject

    if props.created:
        meta["doc_created"] = props.created.isoformat()

    if props.modified:
        meta["doc_modified"] = props.modified.isoformat()

    return meta


def _extract_image_captions(doc: WordDocument) -> list[str]:
    """Extract alt text/descriptions from inline images in DOCX."""
    captions = []

    for p in doc.paragraphs:
        for drawing in xml_element(p).findall(f".//{qn('w:drawing')}"):
            for drawing_properties in drawing.findall(f".//{qn('wp:docPr')}"):
                descr = drawing_properties.get(qn("wp:descr")) or drawing_properties.get("descr", "")

                if descr and descr.strip():
                    captions.append(f"[image: {descr.strip()}]")

        for obj in xml_element(p).findall(f".//{qn('w:object')}"):
            ole = obj.find(f".//{qn('o:OLEObject')}")

            if ole is not None:
                prog_id = ole.get(qn("o:ProgID")) or "OLE object"
                captions.append(f"[embedded: {prog_id}]")

        captions.extend("[diagram: SmartArt]" for _ in xml_element(p).findall(f".//{qn('dgm:relents')}"))

    return captions


def docx_table_to_markdown(table: Table) -> str:
    """Convert a python-docx table to markdown table format."""
    rows = []

    for row in table.rows:
        cells = [markdown_cell(cell.text) for cell in row.cells]
        rows.append(cells)

    return render_table(rows)


# --- parse_docx_sections helpers (extracted to reduce C901) -----------------


@dataclass
class DocxSectionState:
    """Heading and numbering state for the document body walk."""

    sections: list[Section]
    heading_stack: list[tuple[int, str]]
    current_heading: str | None = None
    inferred_heading: bool = True
    styled_level: int = 0
    current_lines: list[str] = field(default_factory=list)
    current_hints: list[dict[str, Any]] = field(default_factory=list)
    current_length: int = 0
    numberer: ListNumberer = field(default_factory=lambda: ListNumberer({}))


MAX_MANUAL_HEADING_CHARS = 120


def paragraph_is_bold(p: Paragraph) -> bool:
    runs = [run for run in p.runs if run.text.strip()]
    return bool(
        runs and all(run.bold or (run.bold is None and p.style and p.style.font.bold) for run in runs)
    )


def heading_style_level(p: Paragraph) -> int | None:
    """Return 0 for Title, 1-N for Heading N (English or Russian style names)."""
    style_name = (p.style.name or "").lower() if p.style else ""

    if style_name in {"title", "название"}:
        return 0

    if match := re.fullmatch(r"(?:heading|заголовок)\s*(\d+)?", style_name):
        return int(match.group(1) or 1)

    return None


def detect_heading_level(p: Paragraph) -> int | None:
    """Styles are authoritative; short emphasized lines provide a fallback.

    Numbering alone is not a heading signal: ordinary short list items would
    otherwise replace the real section path. The fallback level is relative
    to the enclosing styled heading (see ``process_section_paragraph``).
    """
    if (level := heading_style_level(p)) is not None:
        return level

    text = _paragraph_full_text(p).strip()

    if not text or len(text) >= MAX_MANUAL_HEADING_CHARS or text.endswith((".", ":", ";", ",", "!", "?")):
        return None

    if not (paragraph_is_bold(p) or (text.isupper() and any(char.isalpha() for char in text))):
        return None

    list_info = _paragraph_list_info(p)
    return list_info[1] + 1 if list_info else 1


def flush_docx_text(state: DocxSectionState) -> None:
    content = "\n".join(state.current_lines).strip()

    if content:
        state.sections.append(
            Section(
                state.current_heading,
                content,
                metadata={"line_hints": state.current_hints, "inferred_heading": state.inferred_heading},
            )
        )

    state.current_lines = []
    state.current_hints = []
    state.current_length = 0


def apply_heading(state: DocxSectionState, heading_level: int, text: str) -> None:
    """Flush current content and update heading stack."""
    if not state.current_hints or not all(hint.get("heading") for hint in state.current_hints):
        flush_docx_text(state)

    while state.heading_stack and state.heading_stack[-1][0] >= heading_level:
        state.heading_stack.pop()

    state.heading_stack.append((heading_level, text))
    state.current_heading = " > ".join(h for _, h in state.heading_stack)


def flush_table(state: DocxSectionState, md_table: str) -> None:
    """Flush preceding text and append the rendered table."""
    if md_table:
        flush_docx_text(state)

        state.sections.append(Section(state.current_heading, md_table, PageContentType.TABLE))
        state.current_lines = []


def process_section_paragraph(state: DocxSectionState, paragraph: Paragraph, text: str) -> None:
    text = text.strip()

    if not text:
        return

    heading_level = detect_heading_level(paragraph)
    styled = heading_style_level(paragraph) is not None
    list_info = _paragraph_list_info(paragraph)

    if list_info:
        num_id, level = list_info

        if not state.numberer.is_ordered(num_id, level):
            heading_level = None

        text = state.numberer.prefix(num_id, level).strip() + " " + text

    if heading_level is not None and not styled:
        heading_level += state.styled_level

    if heading_level is not None:
        apply_heading(state, heading_level, text)

    if heading_level is not None and styled:
        state.styled_level = heading_level
        state.inferred_heading = False
        return

    if heading_level is not None:
        state.inferred_heading = True

    start = state.current_length
    state.current_length += len(text) + 1
    state.current_hints.append(
        {
            "start": start,
            "end": start + len(text),
            "text": text,
            "bold": paragraph_is_bold(paragraph),
            "heading": heading_level is not None,
            "numbering_level": list_info[1] if list_info else None,
        }
    )
    # Retain heading text in the body as well as the breadcrumb. Numbered titles
    # are semantic markers used by the domain profile and flat-text classifier.
    state.current_lines.append(text)


def _collect_docx_metadata(
    doc: WordDocument,
    page_numbers: list[int],
    paragraph_count: int,
    table_count: int,
    image_count: int,
) -> dict[str, Any]:
    """Build metadata dict from document statistics and properties."""
    metadata: dict[str, Any] = {}

    if page_numbers:
        metadata["page_start"] = page_numbers[0]
        metadata["page_end"] = page_numbers[-1]
        metadata["pages"] = sorted(set(page_numbers))

    metadata["paragraph_count"] = paragraph_count
    metadata["table_count"] = table_count
    metadata["image_count"] = image_count

    headers_footers = _extract_headers_footers(doc)

    if headers_footers["headers"]:
        metadata["header_text"] = "\n".join(headers_footers["headers"])

    if headers_footers["footers"]:
        metadata["footer_text"] = "\n".join(headers_footers["footers"])

    metadata.update(_extract_document_properties(doc))
    return metadata


# --- Public API -------------------------------------------------------------


def append_docx_paragraph(
    paragraph: Paragraph,
    text: str,
    numberer: ListNumberer,
    parts: list[str],
    page_numbers: list[int],
    *,
    current_page: int,
) -> int:
    if _has_page_break(paragraph):
        current_page += 1

    if text.strip():
        page_numbers.append(current_page)
        list_info = _paragraph_list_info(paragraph)

        if list_info:
            num_id, level = list_info
            text = numberer.prefix(num_id, level) + text

        parts.append(text)

    return current_page


def read_docx_body(
    doc: WordDocument,
    formats: dict[tuple[str, int], str],
    level_texts: dict[tuple[str, int], str] | None = None,
) -> tuple[list[str], list[int], int, int, DocxSectionState]:
    numberer = ListNumberer(formats, level_texts)
    parts: list[str] = []
    page_numbers: list[int] = []
    current_page = 1
    paragraph_count = 0
    table_count = 0

    body = doc.element.body
    para_tag = qn("w:p")
    tbl_tag = qn("w:tbl")

    para_by_element = {id(xml_element(p)): p for p in doc.paragraphs}
    tbl_by_element = {id(xml_element(t)): t for t in doc.tables}

    state = DocxSectionState(sections=[], heading_stack=[], numberer=ListNumberer(formats, level_texts))

    for child in body:
        if child.tag == para_tag:
            paragraph_count += 1
            p = para_by_element.get(id(child))

            if p is not None:
                text = _paragraph_full_text(p)
                process_section_paragraph(state, p, text)
                current_page = append_docx_paragraph(
                    p, text, numberer, parts, page_numbers, current_page=current_page
                )
        elif child.tag == tbl_tag:
            table_count += 1
            table = tbl_by_element.get(id(child))

            if table is not None:
                markdown = docx_table_to_markdown(table)
                flush_table(state, markdown)

                if markdown:
                    parts.extend(["", markdown])

    return parts, page_numbers, paragraph_count, table_count, state


def read_docx(file_path: Path) -> tuple[str, list[Section], dict[str, Any]]:
    """Read body order, sections and document metadata in one document pass."""
    validate_docx_archive(file_path)
    doc = docx.Document(str(file_path))
    formats = _get_numbering_formats(doc)
    parts, page_numbers, paragraph_count, table_count, state = read_docx_body(
        doc, formats, get_numbering_texts(doc)
    )

    image_count = sum(1 for _ in doc.element.body.iter(qn("w:drawing")))

    captions = _extract_image_captions(doc)
    parts.extend(captions)

    footnotes = _extract_notes(doc, "footnotes", "w:footnote")

    for i, note in enumerate(footnotes, start=1):
        parts.append(f"[footnote {i}: {note}]")

    endnotes = _extract_notes(doc, "endnotes", "w:endnote")

    for i, note in enumerate(endnotes, start=1):
        parts.append(f"[endnote {i}: {note}]")

    metadata = _collect_docx_metadata(doc, page_numbers, paragraph_count, table_count, image_count)

    flush_docx_text(state)

    note_lines = [f"[footnote {i}: {n}]" for i, n in enumerate(footnotes, start=1)]
    note_lines += [f"[endnote {i}: {n}]" for i, n in enumerate(endnotes, start=1)]

    if note_lines:
        state.sections.append(Section("Footnotes", "\n".join(note_lines)))

    text = "\n".join(parts)
    return text, state.sections or [Section(None, text)], metadata


def parse_docx(file_path: Path) -> tuple[str, dict[str, Any]]:
    """Parse DOCX and return flat text and metadata."""
    text, _sections, metadata = read_docx(file_path)
    return text, metadata


def parse_docx_sections(file_path: Path) -> list[Section]:
    """Split DOCX by Heading/Title styles, preserving table kinds."""
    _text, sections, _metadata = read_docx(file_path)
    return sections
