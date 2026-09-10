"""Content-based text splitting — recursive descent by structural boundaries.

Replaces RecursiveCharacterTextSplitter for structured domains.
Splitting is always by content (regex boundaries), never by char count.
max_unit_chars is a safety-net: triggers descent to finer boundary levels,
not a splitting criterion.
"""

from __future__ import annotations

import re
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
    min_chunk_chars: int = 0,
) -> list[SplitUnit]:
    """Split text by structural boundary hierarchy.

    Recursively descends through boundary levels. If a unit at the current level
    exceeds max_unit_chars, the next finer level is tried on that unit's text.
    The finest level (typically sentence) is the terminal fallback.

    min_chunk_chars: when > 0, units smaller than this are merged with their
    neighbor (same unit_kind, also small) to avoid tiny fragments like "(в ред.".
    Default is 0 (disabled) — enable explicitly in domain splitting paths.
    """
    if not levels:
        return [SplitUnit(None, text.strip(), "raw")] if text.strip() else []

    result = _recursive_split(text, levels, max_unit_chars)
    if min_chunk_chars > 0:
        result = _merge_small_units(result, min_chunk_chars)
    return result


def _recursive_split(text: str, levels: list[BoundaryLevel], max_unit_chars: int) -> list[SplitUnit]:
    """Recursively descend through boundary levels."""
    if not levels:
        return [SplitUnit(None, text.strip(), "raw")] if text.strip() else []

    level, *rest = levels
    raw_units = _split_at_boundary(text, level)

    result: list[SplitUnit] = []
    for unit in raw_units:
        if len(unit.content) > max_unit_chars and rest:
            result.extend(_recursive_split(unit.content, rest, max_unit_chars))
        else:
            result.append(unit)
    return result


def _merge_small_units(units: list[SplitUnit], min_chars: int) -> list[SplitUnit]:
    """Merge tiny orphaned fragments with their neighbor to avoid standalone chunks.

    Three-tier merge logic:
    1. Very tiny units (< 20 chars) that are bare numbers/fragments — absorb into
       the NEXT content-bearing unit (not previous, since previous may be large).
       Catches bare point numbers "2.", "4." that have no real content.
    2. Very tiny units (< 20 chars) — absorb into previous unit (any kind).
       Catches tiny fragments like "(в ред." split off mid-sentence.
    3. Small units (< min_chars) — merge only with same unit_kind, both small.
       Catches sentence fragments split off at abbreviation periods.
    """
    if not units or min_chars <= 0:
        return units

    TINY_THRESHOLD = 20
    _BARE_NUMBER_RE = re.compile(r"^\s*\d+\.\s*$")

    # Pass 1: identify bare number fragments (like "2.", "4.")
    is_bare_number = [
        bool(_BARE_NUMBER_RE.match(u.content)) for u in units
    ]

    # Pass 2: merge — first handle bare numbers (into next), then tiny fragments (into prev)
    merged: list[SplitUnit] = []
    i = 0
    while i < len(units):
        unit = units[i]

        # Tier 1: bare number fragment — absorb into NEXT unit
        if is_bare_number[i] and i + 1 < len(units):
            next_unit = units[i + 1]
            merged_content = unit.content.rstrip() + "\n" + next_unit.content
            merged.append(SplitUnit(
                heading=next_unit.heading,
                content=merged_content,
                unit_kind=next_unit.unit_kind,
                boundary_value=next_unit.boundary_value,
            ))
            i += 2  # skip both
            continue

        # Tier 2: tiny fragment (< 20 chars) — absorb into previous
        if merged and len(unit.content) < TINY_THRESHOLD:
            prev = merged[-1]
            merged[-1] = SplitUnit(
                heading=prev.heading,
                content=prev.content + "\n" + unit.content,
                unit_kind=prev.unit_kind,
                boundary_value=prev.boundary_value,
            )
        # Tier 3: small fragment (< min_chars) — merge only with same kind
        elif (
            merged
            and len(unit.content) < min_chars
            and merged[-1].unit_kind == unit.unit_kind
            and len(merged[-1].content) < min_chars * 2
        ):
            prev = merged[-1]
            merged[-1] = SplitUnit(
                heading=prev.heading,
                content=prev.content + "\n" + unit.content,
                unit_kind=prev.unit_kind,
                boundary_value=prev.boundary_value,
            )
        else:
            merged.append(unit)
        i += 1
    return merged


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
