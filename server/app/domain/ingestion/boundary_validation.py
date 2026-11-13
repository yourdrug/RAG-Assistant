"""Validate structural markers before constructing units; source text stays intact."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import re
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from domain.ingestion.profiles.types import BoundaryLevel


NUMERIC_KINDS = frozenset({"point", "subpoint_num", "subpoint", "clause", "subclause"})
HEADING_KINDS = frozenset({"article", "chapter", "section"})
REFERENCE_WORDS = frozenset({"настоящего", "настоящей", "указанного", "указанной", "кодекса", "закона"})
RUSSIAN_ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
OPTIONAL_LIST_LETTERS = "ёй"
MAX_UNCONFIRMED_NUMBERING_STEP = 2
UPPERCASE_TITLE_MIN_LETTERS = 5
PARENT_HEADING_RE = re.compile(
    r"(?:Статья|Глава|Раздел|Приложение)\s+(?:№\s*)?[\dIVX]+(?:[.-]\d+)*\b",
    re.IGNORECASE,
)
NUMBERING_SCOPE_RE = re.compile(
    r"^[ \t]*(?:Глава|Раздел|Приложение)\s+(?:№\s*)?[\dIVX]+\b", re.MULTILINE | re.IGNORECASE
)


@dataclass(frozen=True)
class BoundaryCandidate:
    level: str
    value: str | None
    start: int
    end: int
    line_start: bool
    prev_char: str
    next_text: str
    format_confirmed: bool = False


def boundary_candidates(
    text: str, level: BoundaryLevel, confirmed_starts: frozenset[int] = frozenset()
) -> list[BoundaryCandidate]:
    result = []

    for match in level.pattern.finditer(text):
        start = match.start() + len(match.group()) - len(match.group().lstrip())
        before = text[:start]
        line_prefix = before.rsplit("\n", 1)[-1]
        result.append(
            BoundaryCandidate(
                level.name,
                match.group(1) if match.groups() else None,
                start,
                match.end(),
                not line_prefix.strip(),
                before.rstrip()[-1:] if before.strip() else "",
                text[match.end() :].split("\n", 1)[0].strip(),
                format_confirmed=start in confirmed_starts,
            )
        )

    return result


def validate_candidates(
    candidates: list[BoundaryCandidate],
    text: str,
    levels: list[BoundaryLevel],
    *,
    parent_number: str | None = None,
    on_event: Callable[[str, int], None] | None = None,
) -> list[BoundaryCandidate]:
    """Reject wrapped references, inconsistent numbering and orphan subpoints.

    Called within one parent scope. A fresh recursive call resets point numbering;
    enclosing chapters/sections naturally reset article numbering. A gap of one
    number is accepted and reported through the optional observation callback.
    Larger forward gaps require a consecutive sibling to confirm the new sequence.
    """
    accepted: list[BoundaryCandidate] = []
    previous_end = 0
    previous: tuple[int, ...] | None = None
    opened = {tuple(int(part) for part in parent_number.split("."))} if parent_number else set()
    siblings: dict[tuple[int, ...], tuple[int, ...]] = {}

    allowed = {level.name for level in levels}
    candidates = [candidate for candidate in candidates if candidate.level in allowed]

    for index, candidate in enumerate(candidates):
        # Flat-format decrees can contain a new numbered chapter or appendix
        # inside one parser section. It opens a fresh parent numbering scope.
        number = candidate_number(candidate)

        # A capitalised title before point 1 starts an appended document
        # (instruction, list of data) with its own numbering.
        if NUMBERING_SCOPE_RE.search(text[previous_end : candidate.start]) or (
            number == (1,) and uppercase_title(prior_line_of(text, candidate.start))
        ):
            previous = None
            siblings.clear()
            opened.clear()

        prior = siblings.get(number[:-1]) if number and candidate.level in NUMERIC_KINDS else previous
        rule = rejection_rule(candidate, text, prior, opened)

        if rule == "number_order" and confirms_forward_sequence(
            candidate, islice(candidates, index + 1, None), text, prior, opened
        ):
            rule = None

        if rule:
            if on_event:
                on_event(rule, candidate.start)

            continue

        if (
            prior
            and number
            and number[:-1] == prior[:-1]
            and numbering_step(candidate, prior) > 1
            and on_event
        ):
            on_event("number_gap", candidate.start)

        if number:
            previous = number
            siblings[number[:-1]] = number
            opened.add(number)

        accepted.append(candidate)
        previous_end = candidate.end

    return accepted


def candidate_number(candidate: BoundaryCandidate) -> tuple[int, ...] | None:
    value = candidate.value or ""

    numeric_pattern = r"\d+(?:[.-]\d+)*" if candidate.level == "article" else r"\d+(?:\.\d+)*"

    if re.fullmatch(numeric_pattern, value):
        return tuple(int(part) for part in re.split(r"[.-]", value))

    if candidate.level == "subpoint" and value and value.lower() in RUSSIAN_ALPHABET:
        return (RUSSIAN_ALPHABET.index(value.lower()),)

    return None


def numbering_step(candidate: BoundaryCandidate, previous: tuple[int, ...]) -> int:
    """Measure list progression, allowing Russian lists to omit ё and й."""
    number = candidate_number(candidate)

    if not number:
        return 0

    step = number[-1] - previous[-1]

    if candidate.level == "subpoint" and (candidate.value or "").lower() in RUSSIAN_ALPHABET:
        step -= sum(
            previous[-1] < RUSSIAN_ALPHABET.index(letter) < number[-1] for letter in OPTIONAL_LIST_LETTERS
        )

    return step


def confirms_forward_sequence(
    candidate: BoundaryCandidate,
    following: Iterable[BoundaryCandidate],
    text: str,
    previous: tuple[int, ...] | None,
    opened: set[tuple[int, ...]],
) -> bool:
    """Recover a forward gap only when the next valid sibling confirms it."""
    number = candidate_number(candidate)

    if (
        not previous
        or not number
        or number[:-1] != previous[:-1]
        or numbering_step(candidate, previous) <= MAX_UNCONFIRMED_NUMBERING_STEP
    ):
        return False

    for sibling in following:
        next_number = candidate_number(sibling)

        # Descendants do not confirm sibling order, but must not hide the next sibling.
        if next_number and len(next_number) > len(number) and next_number[: len(number)] == number:
            continue

        return bool(
            sibling.level == candidate.level
            and next_number
            and next_number[:-1] == number[:-1]
            and numbering_step(sibling, number) == 1
            and not NUMBERING_SCOPE_RE.search(text[candidate.end : sibling.start])
            and not rejection_rule(sibling, text, number, opened)
        )

    return False


def rejection_rule(
    candidate: BoundaryCandidate,
    text: str,
    previous: tuple[int, ...] | None,
    opened: set[tuple[int, ...]],
) -> str | None:
    kind = candidate.level

    if kind not in NUMERIC_KINDS | HEADING_KINDS:
        return None

    return (
        line_rejection(candidate, text)
        or heading_rejection(candidate)
        or numbering_rejection(candidate, previous, opened)
    )


def line_rejection(candidate: BoundaryCandidate, text: str) -> str | None:
    kind = candidate.level
    before = text[: candidate.start].rstrip(" \t")
    prior_line = prior_line_of(text, candidate.start)
    parent_heading = follows_parent_heading(before)

    editorial_note = bool(
        re.fullmatch(
            r"\((?:в ред\.|часть|части|пункт|пп?\.|подпункт|абзац|сноска|преамбула)[^\n]*\)",
            prior_line,
            re.IGNORECASE,
        )
    )

    if not candidate.format_confirmed and (
        not candidate.line_start
        or (
            before.strip()
            and candidate.prev_char not in ".:;!?"
            and not re.search(r"\n[ \t]*\n[ \t]*$", before)
            and not parent_heading
            and not editorial_note
            and not uppercase_title(prior_line)
            and kind in NUMERIC_KINDS
        )
    ):
        return "line_continuation"

    return None


def prior_line_of(text: str, start: int) -> str:
    return text[:start].rstrip().rsplit("\n", 1)[-1].strip()


def uppercase_title(line: str) -> bool:
    """A document or appendix title printed in capitals ends its own line."""
    letters = [char for char in line if char.isalpha()]
    return (
        len(letters) >= UPPERCASE_TITLE_MIN_LETTERS
        and all(char.isupper() for char in letters)
        and not line.endswith((",", ";", "-", "\u2013", "\u2014"))
    )


def follows_parent_heading(before: str) -> bool:
    """Allow a marker immediately after a heading or its separate title line."""
    lines = before.rstrip().rsplit("\n", 2)
    prior_line = lines[-1].strip()

    if PARENT_HEADING_RE.match(prior_line):
        return True

    return bool(
        len(lines) > 1
        and PARENT_HEADING_RE.fullmatch(lines[-2].strip().rstrip(" ."))
        and prior_line
        and (prior_line[0].isupper() or prior_line[0] in '«("')
        and not prior_line.endswith((".", ":", ";", "!", "?"))
    )


def heading_rejection(candidate: BoundaryCandidate) -> str | None:
    if candidate.level in HEADING_KINDS:
        word = candidate.next_text.lstrip(". ").split(maxsplit=1)

        if word and word[0].lower().strip(".,:;") in REFERENCE_WORDS:
            return "heading_reference"

    return None


def numbering_rejection(
    candidate: BoundaryCandidate,
    previous: tuple[int, ...] | None,
    opened: set[tuple[int, ...]],
) -> str | None:
    kind = candidate.level
    number = candidate_number(candidate)

    if kind in {"subpoint_num", "clause", "subclause"} and number and number[:-1] not in opened:
        return "orphan_subpoint"

    if previous and number:
        if kind == "article" and number < previous:
            return "article_order"

        if kind in NUMERIC_KINDS and (
            number[:-1] != previous[:-1]
            or not 1 <= numbering_step(candidate, previous) <= MAX_UNCONFIRMED_NUMBERING_STEP
        ):
            return "number_order"

    return None
