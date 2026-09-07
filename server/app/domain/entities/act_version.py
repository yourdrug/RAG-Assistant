"""ActVersion entity — a specific version of a regulatory act, linked to a document."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


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
