"""Document parsing and classification profile contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from domain.ingestion.profiles.types import (
        BoundaryLevel,
        ConfigDefault,
        EffectiveDateCandidate,
        ReferenceMatch,
    )


@runtime_checkable
class DocumentProfile(Protocol):
    """Strategy object for domain-specific document parsing.

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
        decree header + marker "ПОСТАНОВЛЯЮ" — unmistakable format.
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
