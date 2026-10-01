"""Versioning-related DTOs -- immutable data-transfer objects for act versioning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domain.domain_profile.protocol import ReferenceMatch


@dataclass(frozen=True)
class VersioningResult:
    domain_metadata: dict | None
    act_version_id: int | None
    act_id: int | None
    effective_from: date | None
    warning: str | None


@dataclass(frozen=True)
class VersioningPlan:
    """Extracted version metadata that can be committed with document chunks."""

    extracted_refs: list["ReferenceMatch"]
    domain_metadata: dict | None
    effective_date: date | None
    effective_from: date | None
    date_confidence: float | None
    warning: str | None = None
