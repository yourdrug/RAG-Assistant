"""RegulatoryAct entity — a legal act that may have multiple versions over time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class RegulatoryAct:
    id: int | None
    act_type: str  # DomainProfile.key: 'legal' | 'decree' | ...
    act_number: str | None
    title: str
    issuing_authority: str | None = None
    act_date: date | None = None
    visibility_scope: str = "internal_public"
