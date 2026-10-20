"""Versioning-related DTOs -- immutable data-transfer objects for act versioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict


if TYPE_CHECKING:
    from datetime import date

    from domain.ingestion.profiles.types import ReferenceMatch


@dataclass(frozen=True)
class VersioningResult:
    domain_metadata: dict[str, str | list[str]] | None
    act_version_id: int | None
    act_id: int | None
    effective_from: date | None
    warning: str | None
    effective_to: date | None = None
    is_current: bool = True


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
