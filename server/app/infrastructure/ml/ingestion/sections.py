"""Typed intermediate sections shared by format parsers and preview."""

from dataclasses import dataclass, field, replace
import re
from typing import Any

from domain.knowledge.enums import PageContentType


@dataclass(frozen=True)
class Section:
    heading: str | None
    content: str
    kind: PageContentType = PageContentType.TEXT
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedFormat:
    sections: list[Section]
    metadata: dict[str, Any] = field(default_factory=dict)
    structured: bool = True


TABLE_NOTE_RE = re.compile(r"^(?:примечани[ея]|notes?|сноска|источник|source)\b|^\[\^", re.IGNORECASE)
TABLE_NOTE_ITEM_RE = re.compile(r"^(?:\d+[.)]|[-*•])\s")
# Table introduction: caption and lead-in paragraph always, earlier text up to the limit.
MIN_TABLE_INTRODUCTION_PARAGRAPHS = 2
MAX_TABLE_INTRODUCTION_CHARS = 1000


def table_context_sections(sections: list[Section]) -> list[Section]:
    """Attach adjacent introductions, captions and explicit notes to each table.

    Keep complete text in parent metadata so large conditions survive row
    splitting. Heading and table boundaries prevent inheriting another scope.
    The operation is idempotent for parsers that already attached this context.
    """
    result = []

    for index, section in enumerate(sections):
        if section.kind != PageContentType.TABLE:
            result.append(section)
            continue

        parents = list(section.metadata.get("parent_units", []))
        existing = {parent.get("content") for parent in parents}

        for content in adjacent_table_context(sections, index):
            if content not in existing:
                parents.append(
                    {
                        "heading": section.heading or "",
                        "unit_kind": "table_context",
                        "boundary_value": None,
                        "content": content,
                    }
                )
                existing.add(content)

        result.append(replace(section, metadata={**section.metadata, "parent_units": parents}))

    return result


def adjacent_table_context(sections: list[Section], index: int) -> list[str]:
    context = [table_introduction(sections, index)]

    if index + 1 < len(sections):
        following = sections[index + 1]

        if following.kind == PageContentType.TEXT and following.heading == sections[index].heading:
            context.append(leading_table_notes(following.content.strip()))

    return [content for content in context if content]


def table_introduction(sections: list[Section], index: int) -> str:
    if index == 0:
        return ""

    previous = sections[index - 1]

    if previous.kind != PageContentType.TEXT or previous.heading != sections[index].heading:
        return ""

    introduction = previous.content.strip()

    if (
        index > 1
        and sections[index - 2].kind == PageContentType.TABLE
        and sections[index - 2].heading == sections[index].heading
    ):
        notes = leading_table_notes(introduction)
        introduction = introduction[len(notes) :].strip()

    return trailing_paragraphs(introduction)


def trailing_paragraphs(text: str) -> str:
    """Caption and lead-in before a table: the last paragraphs, not the whole section.

    The last ``MIN_TABLE_INTRODUCTION_PARAGRAPHS`` are kept whole (a large condition
    must survive row splitting); earlier ones only within the character limit.
    A text section without headings can be a whole chapter; copying it into every
    row batch's parent scope made embedding inputs exceed the model limit.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    kept: list[str] = []

    for line in reversed(lines):
        size = sum(len(item) + 1 for item in kept) + len(line)

        if len(kept) >= MIN_TABLE_INTRODUCTION_PARAGRAPHS and size > MAX_TABLE_INTRODUCTION_CHARS:
            break

        kept.insert(0, line)

    return "\n".join(kept).strip()


def leading_table_notes(text: str) -> str:
    """Read consecutive note paragraphs without claiming subsequent prose."""
    end = 0

    for paragraph in re.finditer(r"\S[^\n]*(?:\n(?!\s*\n)[^\n]*)*", text):
        if not TABLE_NOTE_RE.match(paragraph.group()) and not (
            end and TABLE_NOTE_ITEM_RE.match(paragraph.group())
        ):
            break

        end = paragraph.end()

    return text[:end]
