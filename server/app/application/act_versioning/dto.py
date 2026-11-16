"""Versioning-related DTOs -- immutable data-transfer objects for act versioning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

from domain.knowledge.acts.edition_metadata import edition_metadata


if TYPE_CHECKING:
    from datetime import date

    from domain.ingestion.profiles.types import ReferenceMatch
    from domain.knowledge.acts.models import ActVersion


@dataclass(frozen=True)
class VersioningResult:
    domain_metadata: dict[str, str | list[str]] | None = None
    warning: str | None = None
    act_id: int | None = None
    # Edition metadata every chunk of the document carries (empty: not an edition).
    pack_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_version(
        cls, domain_metadata: dict[str, str | list[str]] | None, version: ActVersion
    ) -> VersioningResult:
        return cls(domain_metadata, None, version.act_id, edition_metadata(version))


@dataclass(frozen=True)
class VersioningPlan:
    """Extracted version metadata that can be committed with document chunks."""

    extracted_refs: list[ReferenceMatch]
    domain_metadata: dict[str, str | list[str]] | None
    effective_date: date | None
    effective_from: date | None
    date_confidence: float | None
    warning: str | None = None


@dataclass
class ActSummary:
    id: int
    act_type: str
    act_number: str | None
    title: str
    act_date: date | None
    visibility_scope: str


class VersionUpdateOptions(TypedDict, total=False):
    effective_from: date | None
    effective_to: date | None
    act_id: int | None
    verified_by: int | None
    effective_from_provided: bool
    effective_to_provided: bool
    act_id_provided: bool
