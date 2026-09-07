"""Content-based text splitting — recursive descent by structural boundaries.

Replaces RecursiveCharacterTextSplitter for structured domains.
Splitting is always by content (regex boundaries), never by char count.
max_unit_chars is a safety-net: triggers descent to finer boundary levels,
not a splitting criterion.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.domain_profile.protocol import BoundaryLevel


@dataclass(frozen=True)
class SplitUnit:
    """A single unit after content-based splitting."""

    heading: str | None
    content: str
    unit_kind: str
    boundary_value: str | None = None  # value captured by boundary regex (article number, point number, ...)


def split_by_content(
    text: str,
    levels: list[BoundaryLevel],
    max_unit_chars: int,
) -> list[SplitUnit]:
    """Split text by structural boundary hierarchy.

    Recursively descends through boundary levels. If a unit at the current level
    exceeds max_unit_chars, the next finer level is tried on that unit's text.
    The finest level (typically sentence) is the terminal fallback.
    """
    if not levels:
        return [SplitUnit(None, text.strip(), "raw")] if text.strip() else []

    level, *rest = levels
    raw_units = _split_at_boundary(text, level)

    result: list[SplitUnit] = []
    for unit in raw_units:
        if len(unit.content) > max_unit_chars and rest:
            result.extend(split_by_content(unit.content, rest, max_unit_chars))
        else:
            result.append(unit)
    return result


def _split_at_boundary(text: str, level: BoundaryLevel) -> list[SplitUnit]:
    """Split text at all occurrences of a boundary level's regex pattern."""
    matches = list(level.pattern.finditer(text))
    if not matches:
        return [SplitUnit(None, text.strip(), level.name)] if text.strip() else []

    units: list[SplitUnit] = []

    # Text before the first boundary (preamble/introduction)
    if matches[0].start() > 0:
        lead = text[: matches[0].start()].strip()
        if lead:
            units.append(SplitUnit(None, lead, "preamble"))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        if chunk_text:
            boundary_value = m.group(1) if m.groups() else None
            heading = f"{level.name} {boundary_value}" if boundary_value else level.name
            units.append(SplitUnit(heading, chunk_text, level.name, boundary_value))

    return units
