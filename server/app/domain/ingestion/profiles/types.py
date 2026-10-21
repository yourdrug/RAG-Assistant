"""Data contracts for document classification and parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import date
    import re


@dataclass(frozen=True)
class ConfigDefault:
    """Declarative config parameter default — profile declares what it needs.

    Values are seeded into config_parameters table at startup.
    Profile logic reads them via DomainSettingsPort, never uses these defaults directly.
    """

    key: str
    value: str
    value_type: str  # "int" | "float" | "bool" | "str"
    description: str
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class BoundaryLevel:
    """One level in the structural hierarchy for content-based splitting.

    The pattern matches the START of a new structural unit (chapter, article, etc.).
    Splitting is recursive descent through these levels — not char-count based.

    When ``always_split`` is True the level is applied unconditionally
    (structural boundaries like numeric subpoints).  When False (default)
    the level is only used as a safety-net when the parent unit exceeds
    ``max_unit_chars``.
    """

    name: str  # "chapter" | "article" | "point" | "subpoint" | "sentence"
    pattern: re.Pattern[str]
    always_split: bool = False


@dataclass(frozen=True)
class ReferenceMatch:
    """An extracted reference from document text — goes into chunk metadata."""

    kind: str  # "article" | "point" | "decree_number" | "decree_date" | ...
    value: str


def refs_to_metadata(refs: list[ReferenceMatch]) -> dict[str, str | list[str]]:
    """Convert reference matches into a JSON-serializable metadata dict.

    Repeated kinds collapse into lists so no value is lost:
    [point=1, point=2] -> {"point": ["1", "2"]}.
    """
    metadata: dict[str, str | list[str]] = {}

    for ref in refs:
        existing = metadata.get(ref.kind)

        if existing is None:
            metadata[ref.kind] = ref.value
        elif isinstance(existing, list):
            if ref.value not in existing:
                existing.append(ref.value)
        elif existing != ref.value:
            metadata[ref.kind] = [existing, ref.value]

    return metadata


@dataclass(frozen=True)
class EffectiveDateCandidate:
    """An extracted effective date with confidence scoring."""

    effective_from: date | None = None
    signing_date: date | None = None
    confidence: float = 0.0  # 0..1
