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


def _paragraph_full_text(paragraph) -> str:
    """Full paragraph text, including runs nested inside hyperlinks.

    ``Paragraph.text`` in python-docx only concatenates direct ``<w:r>``
    children of the paragraph. Text inside a ``<w:hyperlink>`` wrapper (i.e.
    any linked text — a very common construct in real documents) lives one
    level deeper in the XML tree and is silently dropped by that property.
    We instead walk the paragraph XML directly and collect every ``<w:t>``
    in document order, so link text is never lost.
    """
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


def _get_numbering_formats(doc) -> dict[tuple[str, int], str]:
    """Map (numId, ilvl) -> numFmt ('bullet', 'decimal', 'lowerLetter', ...).

    Resolved from the document's numbering part (numId -> abstractNumId ->
    per-level numFmt). Returns an empty map if the document has no
    numbering part or it can't be parsed — callers fall back to a plain
    bullet in that case.
    """
    from docx.oxml.ns import qn

    formats: dict[tuple[str, int], str] = {}
    try:
        numbering_part = doc.part.numbering_part
    except Exception:
        return formats
    if numbering_part is None:
        return formats

    root = numbering_part.element
    abstract_formats: dict[str, dict[int, str]] = {}
    for abstract_num in root.findall(qn("w:abstractNum")):
        abs_id = abstract_num.get(qn("w:abstractNumId"))
        lvl_formats: dict[int, str] = {}
        for lvl in abstract_num.findall(qn("w:lvl")):
            ilvl_raw = lvl.get(qn("w:ilvl"))
            if ilvl_raw is None:
                continue
            fmt_el = lvl.find(qn("w:numFmt"))
            if fmt_el is not None:
                lvl_formats[int(ilvl_raw)] = fmt_el.get(qn("w:val"), "bullet")
        if abs_id is not None:
            abstract_formats[abs_id] = lvl_formats

    for num in root.findall(qn("w:num")):
        num_id = num.get(qn("w:numId"))
        abs_ref = num.find(qn("w:abstractNumId"))
        if abs_ref is None or num_id is None:
            continue
        abs_id = abs_ref.get(qn("w:val"))
        for ilvl, fmt in abstract_formats.get(abs_id, {}).items():
            formats[(num_id, ilvl)] = fmt
    return formats


def _paragraph_list_info(paragraph) -> tuple[str, int] | None:
    """Return (numId, ilvl) if paragraph is a list item, else None."""
    from docx.oxml.ns import qn

    pPr = paragraph._element.find(qn("w:pPr"))
    if pPr is None:
        return None
    numPr = pPr.find(qn("w:numPr"))
    if numPr is None:
        return None
    ilvl_el = numPr.find(qn("w:ilvl"))
    numId_el = numPr.find(qn("w:numId"))
    level = int(ilvl_el.get(qn("w:val"), "0")) if ilvl_el is not None else 0
    num_id = numId_el.get(qn("w:val")) if numId_el is not None else None
    if num_id is None:
        return None
    return num_id, level


class _ListNumberer:
    """Assigns sequential numbers to ordered-list items, per (numId, level).

    Mirrors what Word itself renders: a counter resets whenever a deeper
    level starts and resumes where it left off when returning to a
    shallower level within the same list.
    """

    def __init__(self, numbering_formats: dict[tuple[str, int], str]) -> None:
        self._formats = numbering_formats
        self._counters: dict[str, dict[int, int]] = {}

    def prefix(self, num_id: str, level: int) -> str:
        fmt = self._formats.get((num_id, level), "bullet")
        indent = "  " * level
        if fmt == "bullet":
            return f"{indent}- "

        counters = self._counters.setdefault(num_id, {})
        # Starting a new item at this level resets any deeper counters.
        for deeper in [lvl for lvl in counters if lvl > level]:
            del counters[deeper]
        counters[level] = counters.get(level, 0) + 1
        n = counters[level]

        if fmt == "lowerLetter":
            marker = chr(ord("a") + (n - 1) % 26)
        elif fmt == "upperLetter":
            marker = chr(ord("A") + (n - 1) % 26)
        elif fmt in ("lowerRoman", "upperRoman"):
            marker = _to_roman(n)
            if fmt == "lowerRoman":
                marker = marker.lower()
        else:  # decimal and anything else unrecognized
            marker = str(n)
        return f"{indent}{marker}. "


def _to_roman(n: int) -> str:
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
            (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    result = []
    for value, sym in vals:
        count, n = divmod(n, value)
        result.append(sym * count)
    return "".join(result)


def _extract_headers_footers(doc) -> dict:
    """Collect unique header/footer text across all sections.

    python-docx's paragraph iteration (``doc.paragraphs``) only walks the
    main document body — text in ``doc.sections[i].header``/``.footer`` is a
    separate part of the XML tree and is silently skipped. Real documents
    routinely put identifying info there (order number, date, classification
    stamp) that's otherwise lost entirely. A document can have several
    sections with different headers/footers (e.g. a different header from
    page 2 onward); we dedupe identical text across sections since it's
    usually the same header repeated for every section.
    """
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
    """Extract footnote or endnote body text.

    Footnotes/endnotes live in a separate part (footnotes.xml / endnotes.xml)
    linked via a relationship, not exposed through python-docx's public API
    at all — without this they're dropped from every DOCX parse.
    """
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
            continue  # separator / continuation-separator placeholders, not real notes
        note_text = "".join(t.text for t in note.iter(qn("w:t")) if t.text).strip()
        if note_text:
            texts.append(note_text)
    return texts


def _extract_document_properties(doc) -> dict:
    """Pull core document properties (title/author/dates) into metadata.

    Cheap to read and valuable for search facets/filtering; previously
    ignored entirely.
    """
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
    List items get real prefixes: numbered lists render as "1. ", "a. ",
    "i. " etc. matching their actual Word numFmt, bullet lists as "- ".
    Hyperlinked text and image alt text are preserved.
    Headers/footers, footnotes/endnotes, and core document properties
    (title/author/dates) are extracted into metadata rather than dropped.
    """
    doc = docx.Document(str(file_path))
    numberer = _ListNumberer(_get_numbering_formats(doc))
    parts = []
    page_numbers: list[int] = []
    current_page = 1

    for p in doc.paragraphs:
        if _has_page_break(p):
            current_page += 1
        text = _paragraph_full_text(p)
        if not text.strip():
            continue
        page_numbers.append(current_page)

        list_info = _paragraph_list_info(p)
        if list_info:
            num_id, level = list_info
            parts.append(numberer.prefix(num_id, level) + text)
        else:
            parts.append(text)

    for table in doc.tables:
        md_table = docx_table_to_markdown(table)
        if md_table:
            parts.append("")  # blank line before table
            parts.append(md_table)

    # Extract image captions
    captions = _extract_image_captions(doc)
    for cap in captions:
        parts.append(cap)

    # Footnotes/endnotes are real body content, not metadata — surface them
    # as a labeled trailing section so they stay searchable.
    footnotes = _extract_notes(doc, "footnotes", "w:footnote")
    for i, note in enumerate(footnotes, start=1):
        parts.append(f"[footnote {i}: {note}]")
    endnotes = _extract_notes(doc, "endnotes", "w:endnote")
    for i, note in enumerate(endnotes, start=1):
        parts.append(f"[endnote {i}: {note}]")

    metadata: dict = {}
    if page_numbers:
        metadata["page_start"] = page_numbers[0]
        metadata["page_end"] = page_numbers[-1]
        metadata["pages"] = sorted(set(page_numbers))

    headers_footers = _extract_headers_footers(doc)
    if headers_footers["headers"]:
        metadata["header_text"] = "\n".join(headers_footers["headers"])
    if headers_footers["footers"]:
        metadata["footer_text"] = "\n".join(headers_footers["footers"])

    metadata.update(_extract_document_properties(doc))

    return "\n".join(parts), metadata


def parse_docx_sections(file_path: Path) -> list[tuple[str | None, str]]:
    """Split DOCX into (heading, content) sections by Heading*/Title styles.

    The returned heading is a breadcrumb path ("Parent > Child > ...") built
    from the full heading hierarchy leading to that section (by Heading N
    level), so a deeply nested section stays self-describing even without
    its sibling sections for context. List items and hyperlinked text are
    preserved the same way as in parse_docx.
    """
    doc = docx.Document(str(file_path))
    numberer = _ListNumberer(_get_numbering_formats(doc))
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    # Stack of (level, heading_text) tracking the current breadcrumb path.
    heading_stack: list[tuple[int, str]] = []

    def _flush() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_heading, content))

    for p in doc.paragraphs:
        text = _paragraph_full_text(p).strip()
        if not text:
            continue
        style_name = (p.style.name or "").lower() if p.style else ""

        heading_level = None
        if style_name == "title":
            heading_level = 0
        elif style_name.startswith("heading"):
            suffix = style_name.replace("heading", "").strip()
            heading_level = int(suffix) if suffix.isdigit() else 1

        if heading_level is not None:
            _flush()
            while heading_stack and heading_stack[-1][0] >= heading_level:
                heading_stack.pop()
            heading_stack.append((heading_level, text))
            current_heading = " > ".join(h for _, h in heading_stack)
            current_lines = []
        else:
            list_info = _paragraph_list_info(p)
            if list_info:
                num_id, level = list_info
                current_lines.append(numberer.prefix(num_id, level) + text)
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

    # Footnotes/endnotes aren't reachable via doc.paragraphs at all (see
    # _extract_notes), so they need to be appended as their own section here
    # rather than relying on the (unused, for this code path) parse_docx text.
    note_lines = [f"[footnote {i}: {n}]" for i, n in enumerate(_extract_notes(doc, "footnotes", "w:footnote"), start=1)]
    note_lines += [f"[endnote {i}: {n}]" for i, n in enumerate(_extract_notes(doc, "endnotes", "w:endnote"), start=1)]
    if note_lines:
        sections.append(("Footnotes", "\n".join(note_lines)))

    if not sections:
        text, _meta = parse_docx(file_path)
        return [(None, text)]
    return sections
