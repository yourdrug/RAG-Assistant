"""DOCX parsing — python-docx based text/table/section extraction."""

from __future__ import annotations

from pathlib import Path

import docx


def _has_page_break(paragraph) -> bool:
    """Check if a paragraph contains a manual page break."""
    from docx.oxml.ns import qn

    for run in paragraph.runs:
        for br in run._element.findall(qn("w:br")):
            if br.get(qn("w:type")) == "page":
                return True
    return False


def _paragraph_list_prefix(paragraph) -> str | None:
    """Detect if paragraph is a list item and return indent prefix.

    Checks <w:numPr> in paragraph properties XML. Returns a prefix like
    "- " for level 0, "  - " for level 1, etc. Returns None if not a list item.
    """
    from docx.oxml.ns import qn

    pPr = paragraph._element.find(qn("w:pPr"))
    if pPr is None:
        return None
    numPr = pPr.find(qn("w:numPr"))
    if numPr is None:
        return None
    ilvl = numPr.find(qn("w:ilvl"))
    level = int(ilvl.get(qn("w:val"), "0")) if ilvl is not None else 0
    return "  " * level + "- "


def _extract_image_captions(doc) -> list[str]:
    """Extract alt text/descriptions from inline images in DOCX.

    Scans <w:drawing> elements for <wp:docPr descr="..."> attributes.
    """
    from docx.oxml.ns import qn

    captions = []
    for p in doc.paragraphs:
        for drawing in p._element.findall(f".//{qn('w:drawing')}"):
            for docPr in drawing.findall(f".//{qn('wp:docPr')}"):
                descr = docPr.get(qn("wp:descr")) or docPr.get("descr", "")
                if descr and descr.strip():
                    captions.append(f"[image: {descr.strip()}]")
    return captions


def docx_table_to_markdown(table) -> str:
    """Convert a python-docx table to markdown table format.

    First row is used as header. Merged cells are handled by python-docx
    which returns the merged value for each cell in the range.
    """
    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace("|", "\\|") for cell in row.cells]
        rows.append(cells)
    if not rows:
        return ""

    # Normalize column count
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    header = "| " + " | ".join(rows[0]) + " |"
    separator = "|" + "|".join(["---"] * max_cols) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, separator] + body) if body else header


def parse_docx(file_path: Path) -> tuple[str, dict]:
    """Parse DOCX and return (text, metadata).

    Detects page breaks via <w:br w:type="page"/> to provide page metadata.
    Tables are serialized as markdown tables.
    List items get restored prefixes (- ,   - ).
    Image alt text is extracted from inline drawings.
    """
    doc = docx.Document(str(file_path))
    parts = []
    page_numbers: list[int] = []
    current_page = 1

    for p in doc.paragraphs:
        if _has_page_break(p):
            current_page += 1
        if not p.text.strip():
            continue
        page_numbers.append(current_page)

        prefix = _paragraph_list_prefix(p)
        if prefix:
            parts.append(prefix + p.text)
        else:
            parts.append(p.text)

    for table in doc.tables:
        md_table = docx_table_to_markdown(table)
        if md_table:
            parts.append("")  # blank line before table
            parts.append(md_table)

    # Extract image captions
    captions = _extract_image_captions(doc)
    for cap in captions:
        parts.append(cap)

    metadata: dict = {}
    if page_numbers:
        metadata["page_start"] = page_numbers[0]
        metadata["page_end"] = page_numbers[-1]
        metadata["pages"] = sorted(set(page_numbers))
    return "\n".join(parts), metadata


def parse_docx_sections(file_path: Path) -> list[tuple[str | None, str]]:
    """Split DOCX into (heading, content) sections by Heading*/Title styles.

    Also detects page breaks to track page numbers within each section.
    """
    doc = docx.Document(str(file_path))
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_heading, content))

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style_name = (p.style.name or "").lower() if p.style else ""
        if style_name.startswith("heading") or style_name == "title":
            _flush()
            current_heading = text
            current_lines = []
        else:
            current_lines.append(text)
    _flush()

    table_lines = []
    for table in doc.tables:
        md_table = docx_table_to_markdown(table)
        if md_table:
            table_lines.append(md_table)
    if table_lines:
        sections.append((None, "\x00TABLE:" + "\n\n".join(table_lines)))

    if not sections:
        text, _meta = parse_docx(file_path)
        return [(None, text)]
    return sections
