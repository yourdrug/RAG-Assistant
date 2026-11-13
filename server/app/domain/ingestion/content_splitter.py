"""Content-based text splitting — recursive descent by structural boundaries.

Replaces RecursiveCharacterTextSplitter for structured domains.
Text is cut at semantic boundaries. The measured body budget decides when to descend
to a finer level; it never cuts through a structural unit at an arbitrary character.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

from domain.ingestion.boundary_validation import boundary_candidates, validate_candidates
from domain.knowledge.chunk import SENTENCE_UNIT_KIND


if TYPE_CHECKING:
    from collections.abc import Callable

    from domain.ingestion.profiles.types import BoundaryLevel


MIN_STRUCTURAL_CHUNK_CHARS = 100
TINY_FRAGMENT_MAX_CHARS = 20
HEADING_INTRO_MAX_CHARS = 200
HEADING_INTRO_MAX_LINES = 2
ENCLOSING_LEVEL_COUNT = 2


@dataclass(frozen=True)
class BoundaryContext:
    """Identity and introductory condition of a parent structural unit."""

    heading: str
    unit_kind: str
    boundary_value: str | None
    content: str


@dataclass(frozen=True)
class SplitUnit:
    """A single unit after content-based splitting."""

    heading: str | None
    content: str
    unit_kind: str
    boundary_value: str | None = None  # value captured by boundary regex (article number, point number, ...)
    parents: tuple[BoundaryContext, ...] = ()
    start: int = 0
    end: int = 0


def needs_finer_boundary(
    unit: SplitUnit,
    level: BoundaryLevel,
    rest: list[BoundaryLevel],
    budget: int,
    measure: Callable[[str], int] = len,
) -> bool:
    """Respect the body budget and mandatory or otherwise hidden structure."""
    return (
        measure(unit.content) > budget
        or any(child.always_split and child.pattern.search(unit.content) for child in rest)
        or (
            not level.pattern.search(unit.content)
            and any(child.name != SENTENCE_UNIT_KIND and child.pattern.search(unit.content) for child in rest)
        )
    )


def merge_small_units(units: list[SplitUnit], min_chars: int) -> list[SplitUnit]:
    """Attach bare number fragments to the next sibling; packing handles small bodies."""
    if not units or min_chars <= 0:
        return units

    bare_number_re = re.compile(r"^\s*\d+(?:\.\d+)*\.\s*$")

    # Pass 1: identify bare number fragments (like "2.", "4.")
    is_bare_number = [bool(bare_number_re.match(u.content)) for u in units]

    # Attach bare numeric markers to their following body.
    merged: list[SplitUnit] = []
    i = 0

    while i < len(units):
        unit = units[i]

        # Tier 1: bare number fragment — absorb into NEXT unit
        if (
            is_bare_number[i]
            and i + 1 < len(units)
            and unit.parents == units[i + 1].parents
            and unit.boundary_value == units[i + 1].boundary_value
        ):
            next_unit = units[i + 1]
            merged_content = unit.content.rstrip() + "\n" + next_unit.content
            merged.append(
                SplitUnit(
                    heading=next_unit.heading,
                    content=merged_content,
                    unit_kind=next_unit.unit_kind,
                    boundary_value=next_unit.boundary_value,
                    parents=next_unit.parents,
                    start=unit.start,
                    end=next_unit.end,
                )
            )
            i += 2  # skip both
            continue

        merged.append(unit)

        i += 1

    return merged


def split_at_boundary(
    text: str,
    level: BoundaryLevel,
    *,
    unit_label: Callable[[str, str], str] | None = None,
    following_levels: list[BoundaryLevel] | None = None,
    parent_number: str | None = None,
    on_boundary_event: Callable[[str, int], None] | None = None,
    confirmed_starts: frozenset[int] = frozenset(),
) -> list[SplitUnit]:
    """Split text at all occurrences of a boundary level's regex pattern."""
    matches = validate_candidates(
        boundary_candidates(text, level, confirmed_starts),
        text,
        [level, *(following_levels or [])],
        parent_number=parent_number,
        on_event=on_boundary_event,
    )

    if not matches:
        return (
            [
                SplitUnit(
                    None,
                    text.strip(),
                    SENTENCE_UNIT_KIND if level.name == SENTENCE_UNIT_KIND else "preamble",
                    start=len(text) - len(text.lstrip()),
                    end=len(text.rstrip()),
                )
            ]
            if text.strip()
            else []
        )

    units: list[SplitUnit] = []

    # Text before the first boundary (preamble/introduction)
    if matches[0].start > 0:
        lead = text[: matches[0].start].strip()

        if lead:
            units.append(
                SplitUnit(
                    None,
                    lead,
                    "preamble",
                    start=len(text[: matches[0].start]) - len(text[: matches[0].start].lstrip()),
                    end=len(text[: matches[0].start].rstrip()),
                )
            )

    for i, m in enumerate(matches):
        start = m.start
        end = matches[i + 1].start if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()

        if chunk_text:
            boundary_value = m.value
            heading = boundary_heading(chunk_text, level, boundary_value, unit_label, following_levels or [])
            units.append(
                SplitUnit(
                    heading,
                    chunk_text,
                    level.name,
                    boundary_value,
                    start=start + len(text[start:end]) - len(text[start:end].lstrip()),
                    end=start + len(text[start:end].rstrip()),
                )
            )

    return units


def boundary_heading(
    text: str,
    level: BoundaryLevel,
    value: str | None,
    unit_label: Callable[[str, str], str] | None,
    following_levels: list[BoundaryLevel],
) -> str | None:
    """Keep actual enclosing titles and short labels for numbered prose units."""
    if not value or level.name == SENTENCE_UNIT_KIND:
        return None

    if level.packing_barrier:
        lines = text.splitlines()
        heading = lines[0].strip()
        marker = level.pattern.match(heading)

        # A chapter title is often printed on the line after its number.
        if marker and not heading[marker.end() :].strip(" .") and len(lines) > 1:
            title = lines[1].strip()

            if (
                title
                and len(title) <= HEADING_INTRO_MAX_CHARS
                and not title.endswith((".", ":", ";"))
                and not any(child.pattern.match(title) for child in following_levels)
            ):
                heading += " " + title

        heading = " ".join(heading.split())

        # Body text on the marker line is not a title; keep only the short label.
        if len(heading) > HEADING_INTRO_MAX_CHARS:
            label = unit_label(level.name, value) if unit_label else ""
            return label or (" ".join(marker.group(0).split()) if marker else value)

        return heading

    return (unit_label(level.name, value) if unit_label else value) or None


def heading_only_intro(text: str, level: BoundaryLevel) -> bool:
    """Recognize short heading blocks and dependent list lead-ins, never prose sentences."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(text) > HEADING_INTRO_MAX_CHARS or not lines:
        return False

    if len(lines) == 1 and text.endswith(":"):
        return True

    return (
        len(lines) <= HEADING_INTRO_MAX_LINES
        and not text.endswith((".", "!", "?", ";"))
        and level.packing_barrier
        and bool(level.pattern.match(lines[0]))
    )


def order_enclosing_levels(text: str, levels: list[BoundaryLevel]) -> list[BoundaryLevel]:
    """Order profile-designated enclosing levels by their first occurrence."""
    if len(levels) >= ENCLOSING_LEVEL_COUNT and all(level.order_by_position for level in levels[:2]):
        first, second = (level.pattern.search(text) for level in levels[:2])

        if second is not None and (first is None or second.start() < first.start()):
            levels = [levels[1], levels[0], *levels[2:]]

    return levels


def unit_parent_context(unit: SplitUnit, rest: list[BoundaryLevel]) -> BoundaryContext:
    """Retain the enclosing heading and condition before descending to children."""
    # Keep the condition before the first finer structural boundary.
    # For sentence descent, only the heading line is context; the
    # complete body stays in the children and is never duplicated.
    parent_number = unit.boundary_value if unit.unit_kind in ("point", "subpoint_num") else None
    starts = [
        candidate.start
        for child_level in rest
        if child_level.name != SENTENCE_UNIT_KIND
        for candidate in validate_candidates(
            boundary_candidates(unit.content, child_level),
            unit.content,
            rest,
            parent_number=parent_number,
        )[:1]
    ]
    intro = unit.content[: min(starts)].strip() if starts else unit.content.split("\n", 1)[0]

    return BoundaryContext(unit.heading or "", unit.unit_kind, unit.boundary_value, intro)
