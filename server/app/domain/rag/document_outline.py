"""Whether a numbered article or table the question cites exists in the act it names.

A missed search is not proof of absence: only the act's own list of articles or
tables, read from all of its chunks, can show that «статья 40» is not there.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from domain.rag.classification import extract_structural_references


TABLE_REFERENCE_RE = re.compile(r"\bтаблиц\w*\s+(?:N|№)?\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^\s*Таблица\s+(\d+(?:\.\d+)*)", re.IGNORECASE)
# kind → (plural nominative, also singular genitive; plural genitive)
UNIT_NOUNS = {"article": ("статьи", "статей"), "table": ("таблицы", "таблиц")}


@dataclass(frozen=True)
class CitedUnit:
    kind: str  # "article" | "table"
    value: str


@dataclass(frozen=True)
class DocumentOutline:
    """Article numbers and table numbers held by one document, under the reader's access."""

    articles: tuple[str, ...]
    tables: tuple[str, ...]


def cited_units(question: str) -> list[CitedUnit]:
    """Articles and tables the question cites by number, in order of appearance."""
    units = [
        CitedUnit("article", ref.value)
        for ref in extract_structural_references(question)
        if ref.kind == "article"
    ]
    units.extend(CitedUnit("table", match[1]) for match in TABLE_REFERENCE_RE.finditer(question))
    return list(dict.fromkeys(units))


def caption_table_number(heading: str | None) -> str | None:
    match = TABLE_CAPTION_RE.match(heading or "")
    return match[1] if match else None


def unit_sort_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.split(r"[.-]", value) if part.isdigit())


def held_units(outline: DocumentOutline, unit: CitedUnit) -> list[str]:
    """Units of the cited kind; tables only of the cited group («4.8» → tables «4.x»)."""
    if unit.kind == "article":
        return sorted(set(outline.articles), key=unit_sort_key)

    prefix = unit.value.rpartition(".")[0]
    tables = {table for table in outline.tables if table.rpartition(".")[0] == prefix}
    return sorted(tables, key=unit_sort_key)


def compact_numbers(values: list[str]) -> str:
    """«1, 2, 3, 5» → «1-3, 5» (with an en dash); numbers with suffixes («26-1») stay separate."""
    parts: list[str] = []
    run: list[str] = []

    def close() -> None:
        if run:
            parts.append(run[0] if len(run) == 1 else f"{run[0]}\N{EN DASH}{run[-1]}")
            run.clear()

    for value in values:
        key = unit_sort_key(value)
        previous = unit_sort_key(run[-1]) if run else None
        simple = "-" not in value

        if simple and previous is not None and key[:-1] == previous[:-1] and key[-1] == previous[-1] + 1:
            run.append(value)
            continue

        close()

        if simple:
            run.append(value)
        else:
            parts.append(value)

    close()
    return ", ".join(parts)


def missing_unit_note(unit: CitedUnit, outline: DocumentOutline, title: str) -> str | None:
    """A context note stating the cited unit is absent, or None when it exists or the list is unknown."""
    held = held_units(outline, unit)

    if not held or unit.value in held:
        return None

    plural, genitive = UNIT_NOUNS[unit.kind]
    return (
        f"Перечень {genitive} документа «{title}» по всему его тексту: {plural} {compact_numbers(held)}. "
        f"{plural.capitalize()} {unit.value} в этом документе нет."
    )
