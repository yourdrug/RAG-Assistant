"""Knowledge domain: models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain.shared.exceptions import ValidationError


if TYPE_CHECKING:
    from datetime import date, datetime


@dataclass
class RegulatoryAct:
    id: int | None
    act_type: str  # DomainProfile.key: 'legal' | 'decree' | ...
    act_number: str | None
    title: str
    issuing_authority: str | None = None
    act_date: date | None = None
    visibility_scope: str = "internal_public"


@dataclass
class ActVersion:
    id: int | None
    act_id: int | None  # FK to RegulatoryAct (NULL = pending manual linkage)
    document_id: int
    effective_from: date | None = None
    effective_to: date | None = None
    is_current: bool = True
    date_source: str = "extracted"  # 'extracted' | 'extracted_trusted' | 'manual'
    date_confidence: float | None = None
    verified_by: int | None = None
    verified_at: datetime | None = None

    def validate_dates(self, *, explicit_end: date | None = None) -> None:
        """Empty superseded editions are valid; an explicit expiry must be later."""
        if self.effective_from is None:
            return

        if explicit_end is not None and explicit_end <= self.effective_from:
            msg = "effective_to must be later than effective_from"
            raise ValidationError(msg)

        if self.effective_to is not None and self.effective_to < self.effective_from:
            msg = "effective_to must not precede effective_from"
            raise ValidationError(msg)
