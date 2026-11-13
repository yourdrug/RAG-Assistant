"""DOCX numbering format resolution — list detection and sequential numbering.

Extracted from ``docx.py`` to reduce its size and isolate the numbering concern.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, cast

from docx.oxml.ns import qn


if TYPE_CHECKING:
    from docx.document import Document
    from docx.oxml.xmlchemy import BaseOxmlElement
    from docx.text.paragraph import Paragraph


def _parse_abstract_formats(root: BaseOxmlElement) -> dict[str, dict[int, str]]:
    """Extract abstract numbering formats: abs_id -> {level: numFmt}."""
    return parse_abstract_values(root, "w:numFmt")


def parse_abstract_values(root: BaseOxmlElement, tag: str) -> dict[str, dict[int, str]]:
    """Extract one per-level ``w:val`` (numFmt, lvlText) for every abstract numbering."""
    abstract_values: dict[str, dict[int, str]] = {}

    for abstract_num in root.findall(qn("w:abstractNum")):
        abs_id = abstract_num.get(qn("w:abstractNumId"))

        if abs_id is None:
            continue

        values: dict[int, str] = {}

        for lvl in abstract_num.findall(qn("w:lvl")):
            ilvl_raw = lvl.get(qn("w:ilvl"))
            element = lvl.find(qn(tag))

            if ilvl_raw is not None and element is not None and element.get(qn("w:val")) is not None:
                values[int(ilvl_raw)] = str(element.get(qn("w:val")))

        abstract_values[abs_id] = values

    return abstract_values


def _resolve_num_formats(
    root: BaseOxmlElement, abstract_formats: dict[str, dict[int, str]]
) -> dict[tuple[str, int], str]:
    """Resolve (numId, ilvl) -> numFmt from w:num elements."""
    formats: dict[tuple[str, int], str] = {}

    for num in root.findall(qn("w:num")):
        num_id = num.get(qn("w:numId"))
        abs_ref = num.find(qn("w:abstractNumId"))

        if abs_ref is None or num_id is None:
            continue

        abs_id = abs_ref.get(qn("w:val"))

        for ilvl, fmt in abstract_formats.get(abs_id or "", {}).items():
            formats[(num_id, ilvl)] = fmt

    return formats


def _get_numbering_formats(doc: Document) -> dict[tuple[str, int], str]:
    """Map (numId, ilvl) -> numFmt ('bullet', 'decimal', 'lowerLetter', ...)."""
    return numbering_values(doc, "w:numFmt")


def get_numbering_texts(doc: Document) -> dict[tuple[str, int], str]:
    """Map (numId, ilvl) -> lvlText template such as ``%1.%2.``."""
    return numbering_values(doc, "w:lvlText")


def numbering_values(doc: Document, tag: str) -> dict[tuple[str, int], str]:
    try:
        numbering_part = doc.part.numbering_part
    except Exception:
        logging.getLogger(__name__).debug("DOCX numbering unavailable", exc_info=True)
        return {}

    if numbering_part is None:
        return {}

    root = numbering_part.element
    return _resolve_num_formats(root, parse_abstract_values(root, tag))


def _paragraph_list_info(paragraph: Paragraph) -> tuple[str, int] | None:
    """Return (numId, ilvl) if paragraph is a list item, else None."""
    paragraph_properties = xml_element(paragraph).find(qn("w:pPr"))

    if paragraph_properties is None:
        return None

    number_properties = paragraph_properties.find(qn("w:numPr"))

    if number_properties is None:
        return None

    ilvl_el = number_properties.find(qn("w:ilvl"))
    number_id_element = number_properties.find(qn("w:numId"))
    level = int(ilvl_el.get(qn("w:val"), "0")) if ilvl_el is not None else 0
    num_id = number_id_element.get(qn("w:val")) if number_id_element is not None else None

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


class ListNumberer:
    """Assigns sequential numbers to ordered-list items, per (numId, level)."""

    def __init__(
        self,
        numbering_formats: dict[tuple[str, int], str],
        level_texts: dict[tuple[str, int], str] | None = None,
    ) -> None:
        self._formats = numbering_formats
        self._texts = level_texts or {}
        self._counters: dict[str, dict[int, int]] = {}

    def is_ordered(self, num_id: str, level: int) -> bool:
        return self._formats.get((num_id, level), "bullet") != "bullet"

    def prefix(self, num_id: str, level: int) -> str:
        fmt = self._formats.get((num_id, level), "bullet")
        indent = "  " * level

        if fmt == "bullet":
            return f"{indent}- "

        counters = self._counters.setdefault(num_id, {})

        for deeper in [lvl for lvl in counters if lvl > level]:
            del counters[deeper]

        counters[level] = counters.get(level, 0) + 1
        template = self._texts.get((num_id, level), "")

        # Multi-level lists ("%1.%2.") must keep parent numbers: contract clauses
        # such as 2.1 are recognised only from the full rendered number.
        if "%" in template:
            label = re.sub(
                r"%(\d)",
                lambda match: list_marker(
                    self._formats.get((num_id, int(match.group(1)) - 1), "decimal"),
                    counters.get(int(match.group(1)) - 1, 1),
                ),
                template,
            ).strip()
        else:
            label = f"{list_marker(fmt, counters[level])}."

        return f"{indent}{label} "


def list_marker(fmt: str, n: int) -> str:
    if fmt == "lowerLetter":
        return chr(ord("a") + (n - 1) % 26)

    if fmt == "upperLetter":
        return chr(ord("A") + (n - 1) % 26)

    if fmt in ("lowerRoman", "upperRoman"):
        marker = _to_roman(n)
        return marker.lower() if fmt == "lowerRoman" else marker

    return str(n)


def xml_element(value: object) -> BaseOxmlElement:
    """Narrow python-docx's XML wrapper at the library boundary."""
    attribute = "_element"
    return cast("BaseOxmlElement", getattr(value, attribute))
