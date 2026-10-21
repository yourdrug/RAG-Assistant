"""Typed intermediate sections shared by format parsers and preview."""

from dataclasses import dataclass, field
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
