"""DomainProfile protocol — strategy object for domain-specific behavior.

A DomainProfile encapsulates everything a domain needs:
- Configuration defaults (seeded into config_parameters)
- Document classification (fingerprint + density score)
- Content-based splitting (structural boundary hierarchy)
- Reference extraction for metadata
- Prompt addenda for RAG generation
- Retrieval policy (fallback to full corpus)

Adding a new domain = new profile file + one registry.register() call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

SENTENCE_UNIT_KIND = "sentence"


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


@runtime_checkable
class DomainProfile(Protocol):
    """Strategy object for domain-specific RAG behavior.

    Profiles are stateless strategy objects (no DB access).
    They receive DomainSettingsPort for reading dynamic config.
    """

    key: str
    display_name: str
    is_versioned: bool

    def config_defaults(self) -> list[ConfigDefault]:
        """Declare config parameters this domain needs.

        Values are seeded into config_parameters at startup.
        Profile logic reads them via DomainSettingsPort — never uses defaults directly.
        """
        ...

    def structural_fingerprint(self, text: str) -> bool:
        """Deterministic format detection — bypasses density scoring.

        True = this document is definitely this domain. Example: a decree
        header "УКАЗ" + marker "ПОСТАНОВЛЯЮ" — unmistakable format.
        """
        ...

    def classify_score(self, text: str) -> float:
        """Density of thematic markers — used when fingerprint doesn't fire."""
        ...

    def content_boundaries(self) -> list[BoundaryLevel]:
        """Structural hierarchy for splitting, from coarsest to finest.

        Empty list = domain has no reliable structure, use char-based fallback.
        """
        ...

    def extract_references(self, text: str) -> list[ReferenceMatch]:
        """Extract structural references for chunk metadata and citation verification."""
        ...

    def extract_effective_date(self, text: str) -> EffectiveDateCandidate | None:
        """Extract effective date with context-bound confidence scoring."""
        ...

    def prompt_addendum(self, breadth: str, as_of_date: date | None = None) -> str | None:
        """Additional prompt rules for this domain.

        For versioned domains with active as_of_date, adds temporal context rules.
        """
        ...

    def retrieval_fallback_to_full_corpus(self) -> bool:
        """Whether to fall back to full corpus when domain-filtered search returns 0 results."""
        ...
