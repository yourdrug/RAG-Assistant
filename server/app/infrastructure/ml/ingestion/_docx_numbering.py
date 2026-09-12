"""DOCX numbering format resolution — list detection and sequential numbering.

Extracted from ``docx.py`` to reduce its size and isolate the numbering concern.
"""

from __future__ import annotations


def _parse_abstract_formats(root) -> dict[str, dict[int, str]]:
    """Extract abstract numbering formats: abs_id -> {level: numFmt}."""
    from docx.oxml.ns import qn

    abstract_formats: dict[str, dict[int, str]] = {}
    for abstract_num in root.findall(qn("w:abstractNum")):
        abs_id = abstract_num.get(qn("w:abstractNumId"))
        if abs_id is None:
            continue
        lvl_formats: dict[int, str] = {}
        for lvl in abstract_num.findall(qn("w:lvl")):
            ilvl_raw = lvl.get(qn("w:ilvl"))
            if ilvl_raw is None:
                continue
            fmt_el = lvl.find(qn("w:numFmt"))
            if fmt_el is not None:
                lvl_formats[int(ilvl_raw)] = fmt_el.get(qn("w:val"), "bullet")
        abstract_formats[abs_id] = lvl_formats
    return abstract_formats


def _resolve_num_formats(root, abstract_formats: dict[str, dict[int, str]]) -> dict[tuple[str, int], str]:
    """Resolve (numId, ilvl) -> numFmt from w:num elements."""
    from docx.oxml.ns import qn

    formats: dict[tuple[str, int], str] = {}
    for num in root.findall(qn("w:num")):
        num_id = num.get(qn("w:numId"))
        abs_ref = num.find(qn("w:abstractNumId"))
        if abs_ref is None or num_id is None:
            continue
        abs_id = abs_ref.get(qn("w:val"))
        for ilvl, fmt in abstract_formats.get(abs_id, {}).items():
            formats[(num_id, ilvl)] = fmt
    return formats


def _get_numbering_formats(doc) -> dict[tuple[str, int], str]:
    """Map (numId, ilvl) -> numFmt ('bullet', 'decimal', 'lowerLetter', ...)."""
    formats: dict[tuple[str, int], str] = {}
    try:
        numbering_part = doc.part.numbering_part
    except Exception:
        return formats
    if numbering_part is None:
        return formats

    root = numbering_part.element
    abstract_formats = _parse_abstract_formats(root)
    return _resolve_num_formats(root, abstract_formats)


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


def _to_roman(n: int) -> str:
    vals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    result = []
    for value, sym in vals:
        count, n = divmod(n, value)
        result.append(sym * count)
    return "".join(result)


class _ListNumberer:
    """Assigns sequential numbers to ordered-list items, per (numId, level)."""

    def __init__(self, numbering_formats: dict[tuple[str, int], str]) -> None:
        self._formats = numbering_formats
        self._counters: dict[str, dict[int, int]] = {}

    def prefix(self, num_id: str, level: int) -> str:
        fmt = self._formats.get((num_id, level), "bullet")
        indent = "  " * level
        if fmt == "bullet":
            return f"{indent}- "

        counters = self._counters.setdefault(num_id, {})
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
        else:
            marker = str(n)
        return f"{indent}{marker}. "
