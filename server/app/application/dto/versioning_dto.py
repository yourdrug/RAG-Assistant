"""Versioning-related DTOs -- immutable data-transfer objects for act versioning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class VersioningResult:
    domain_metadata: dict | None
    act_version_id: int | None
    act_id: int | None
    effective_from: date | None
    warning: str | None
