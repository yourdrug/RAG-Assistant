"""DOCX parsing — python-docx based text/table/section extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import docx


def _has_page_break(paragraph) -> bool:
    """Check if a paragraph contains a manual page break."""
    from docx.oxml.ns import qn

    for run in paragraph.runs:
        for br in run._element.findall(qn("w:br")):
            if br.get(qn("w:type")) == "page":
                return True
    return False


def _paragraph_full_text(paragraph) -> str:
    """Full paragraph text, including runs nested inside hyperlinks."""
    from docx.oxml.ns import qn

    t_tag = qn("w:t")
    tab_tag = qn("w:tab")
    br_tag = qn("w:br")
    parts: list[str] = []
    for node in paragraph._element.iter():
        if node.tag == t_tag:
            if node.text:
                parts.append(node.text)
        elif node.tag == tab_tag:
            parts.append("\t")
        elif node.tag == br_tag:
            parts.append("\n")
    return "".join(parts)


# --- Numbering format resolution — delegated to _docx_numbering ---

from infrastructure.ml.ingestion.docx_numbering import (  # noqa: E402, F401
    _get_numbering_formats,
    ListNumberer,
    _paragraph_list_info,
    _to_roman,
)


# --- Extraction helpers -----------------------------------------------------


def _extract_headers_footers(doc) -> dict:
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
            if container is None or container.is_linked_to_previous:
                continue
            text = "\n".join(p.text.strip() for p in container.paragraphs if p.text.strip())
            if text and text not in seen:
                seen.add(text)
                bucket.append(text)

    return {"headers": headers, "footers": footers}


def _extract_notes(doc, part_reltype_substring: str, xml_tag: str) -> list[str]:
    """Extract footnote or endnote body text."""
    from docx.oxml.ns import qn

    notes_part = None
    for rel in doc.part.rels.values():
        if part_reltype_substring in rel.reltype:
            notes_part = rel.target_part
            break
    if notes_part is None:
        return []

    texts = []
    for note in notes_part.element.findall(qn(xml_tag)):
        note_id = note.get(qn("w:id"))
        if note_id in ("-1", "0"):
            continue
        note_text = "".join(t.text for t in note.iter(qn("w:t")) if t.text).strip()
        if note_text:
            texts.append(note_text)
    return texts


def _extract_document_properties(doc) -> dict:
    """Pull core document properties (title/author/dates) into metadata."""
    props = doc.core_properties
    meta: dict = {}
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


def _extract_image_captions(doc) -> list[str]:
    """Extract alt text/descriptions from inline images in DOCX."""
    from docx.oxml.ns import qn

    captions = []
    for p in doc.paragraphs:
        for drawing in p._element.findall(f".//{qn('w:drawing')}"):
            for docPr in drawing.findall(f".//{qn('wp:docPr')}"):
                descr = docPr.get(qn("wp:descr")) or docPr.get("descr", "")
                if descr and descr.strip():
                    captions.append(f"[image: {descr.strip()}]")

        for obj in p._element.findall(f".//{qn('w:object')}"):
            ole = obj.find(f".//{qn('o:OLEObject')}")
            if ole is not None:
                prog_id = ole.get(qn("o:ProgID")) or "OLE object"
                captions.append(f"[embedded: {prog_id}]")

        for _smart_art in p._element.findall(f".//{qn('dgm:relents')}"):
            captions.append("[diagram: SmartArt]")

    return captions


def docx_table_to_markdown(table) -> str:
    """Convert a python-docx table to markdown table format."""
    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace("|", "\\|") for cell in row.cells]
        rows.append(cells)
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


# --- parse_docx helpers (extracted to reduce C901) --------------------------


def _process_docx_paragraph(
    p,
    numberer: ListNumberer,
    parts: list[str],
    page_numbers: list[int],
    current_page: int,
) -> tuple[int, bool]:
    """Process a single paragraph for parse_docx. Returns (new_page, should_break)."""
    if _has_page_break(p):
        current_page += 1
    text = _paragraph_full_text(p)
    if not text.strip():
        return current_page, True
    page_numbers.append(current_page)
    list_info = _paragraph_list_info(p)
    if list_info:
        num_id, level = list_info
        parts.append(numberer.prefix(num_id, level) + text)
    else:
        parts.append(text)
    return current_page, False


def _process_docx_table(table, parts: list[str]) -> None:
    """Process a single table for parse_docx."""
    md_table = docx_table_to_markdown(table)
    if md_table:
        parts.append("")
        parts.append(md_table)


def _count_images(body, qn) -> int:
    """Count inline images in the document body."""
    return sum(1 for _ in body.iter(qn("w:drawing")))


# --- parse_docx_sections helpers (extracted to reduce C901) -----------------


@dataclass
class DocxSectionState:
    """Mutable state for _process_section_child."""

    sections: list[tuple[str | None, str]]
    heading_stack: list[tuple[int, str]]
    current_heading: str | None = None
    current_lines: list[str] = field(default_factory=list)
    numberer: ListNumberer = field(default_factory=lambda: ListNumberer({}))
    table_map: dict = field(default_factory=dict)


def _detect_heading_level(p) -> int | None:
    """Detect heading level from paragraph style. Returns 0 for Title, 1-N for Heading N, None otherwise."""
    style_name = (p.style.name or "").lower() if p.style else ""
    if style_name == "title":
        return 0
    if style_name.startswith("heading"):
        suffix = style_name.replace("heading", "").strip()
        return int(suffix) if suffix.isdigit() else 1
    return None


def _apply_heading(state: DocxSectionState, heading_level: int, text: str) -> None:
    """Flush current content and update heading stack."""
    content = "\n".join(state.current_lines).strip()
    if content:
        state.sections.append((state.current_heading, content))
    while state.heading_stack and state.heading_stack[-1][0] >= heading_level:
        state.heading_stack.pop()
    state.heading_stack.append((heading_level, text))
    state.current_heading = " > ".join(h for _, h in state.heading_stack)
    state.current_lines = []


def _flush_table(state: DocxSectionState, table) -> None:
    """Convert table to markdown and append as a section."""
    md_table = docx_table_to_markdown(table)
    if md_table:
        content = "\n".join(state.current_lines).strip()
        if content:
            state.sections.append((state.current_heading, content))
        state.sections.append((state.current_heading, "\x00TABLE:" + md_table))
        state.current_lines = []


def _process_section_child(
    child,
    state: DocxSectionState,
    para_by_element: dict[int, Any],
    tbl_by_element: dict[int, Any],
    para_tag: str,
    tbl_tag: str,
) -> None:
    """Process a single body child for parse_docx_sections.

    Mutates *state* in place.
    """
    if child.tag == para_tag:
        p = para_by_element.get(id(child))
        if p is not None:
            text = _paragraph_full_text(p).strip()
            if text:
                heading_level = _detect_heading_level(p)
                if heading_level is not None:
                    _apply_heading(state, heading_level, text)
                else:
                    list_info = _paragraph_list_info(p)
                    if list_info:
                        num_id, level = list_info
                        state.current_lines.append(state.numberer.prefix(num_id, level) + text)
                    else:
                        state.current_lines.append(text)
    elif child.tag == tbl_tag:
        table = tbl_by_element.get(id(child))
        if table:
            _flush_table(state, table)


def _collect_docx_metadata(
    doc,
    page_numbers: list[int],
    paragraph_count: int,
    table_count: int,
    image_count: int,
) -> dict:
    """Build metadata dict from document statistics and properties."""
    metadata: dict = {}
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


def parse_docx(file_path: Path) -> tuple[str, dict]:
    """Parse DOCX and return (text, metadata)."""
    from docx.oxml.ns import qn

    doc = docx.Document(str(file_path))
    numberer = ListNumberer(_get_numbering_formats(doc))
    parts: list[str] = []
    page_numbers: list[int] = []
    current_page = 1
    paragraph_count = 0
    table_count = 0

    body = doc.element.body
    para_tag = qn("w:p")
    tbl_tag = qn("w:tbl")

    para_by_element = {id(p._element): p for p in doc.paragraphs}
    tbl_by_element = {id(t._element): t for t in doc.tables}

    for child in body:
        if child.tag == para_tag:
            paragraph_count += 1
            p = para_by_element.get(id(child))
            if p is not None:
                current_page, _did_break = _process_docx_paragraph(
                    p,
                    numberer,
                    parts,
                    page_numbers,
                    current_page,
                )
        elif child.tag == tbl_tag:
            table_count += 1
            table = tbl_by_element.get(id(child))
            if table is not None:
                _process_docx_table(table, parts)

    image_count = _count_images(body, qn)

    captions = _extract_image_captions(doc)
    for cap in captions:
        parts.append(cap)

    footnotes = _extract_notes(doc, "footnotes", "w:footnote")
    for i, note in enumerate(footnotes, start=1):
        parts.append(f"[footnote {i}: {note}]")
    endnotes = _extract_notes(doc, "endnotes", "w:endnote")
    for i, note in enumerate(endnotes, start=1):
        parts.append(f"[endnote {i}: {note}]")

    metadata = _collect_docx_metadata(doc, page_numbers, paragraph_count, table_count, image_count)

    return "\n".join(parts), metadata


def parse_docx_sections(file_path: Path) -> list[tuple[str | None, str]]:
    """Split DOCX into (heading, content) sections by Heading*/Title styles."""
    from docx.oxml.ns import qn

    doc = docx.Document(str(file_path))

    para_by_element = {id(p._element): p for p in doc.paragraphs}
    tbl_by_element = {id(t._element): t for t in doc.tables}

    state = DocxSectionState(
        sections=[],
        heading_stack=[],
        current_heading=None,
        current_lines=[],
        numberer=ListNumberer(_get_numbering_formats(doc)),
        table_map=tbl_by_element,
    )

    body = doc.element.body
    para_tag = qn("w:p")
    tbl_tag = qn("w:tbl")

    for child in body:
        _process_section_child(child, state, para_by_element, tbl_by_element, para_tag, tbl_tag)

    content = "\n".join(state.current_lines).strip()
    if content:
        state.sections.append((state.current_heading, content))

    fn_notes = _extract_notes(doc, "footnotes", "w:footnote")
    en_notes = _extract_notes(doc, "endnotes", "w:endnote")
    note_lines = [f"[footnote {i}: {n}]" for i, n in enumerate(fn_notes, start=1)]
    note_lines += [f"[endnote {i}: {n}]" for i, n in enumerate(en_notes, start=1)]
    if note_lines:
        state.sections.append(("Footnotes", "\n".join(note_lines)))

    if not state.sections:
        text, _meta = parse_docx(file_path)
        return [(None, text)]
    return state.sections
