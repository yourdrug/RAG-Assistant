"""Legacy preview section DTO, independent of tree adapters."""

from dataclasses import dataclass, field
from typing import Any

from domain.knowledge.enums import PageContentType


@dataclass(frozen=True)
class Section:
    heading: str | None
    content: str
    kind: PageContentType = PageContentType.TEXT
    metadata: dict[str, Any] = field(default_factory=dict)
