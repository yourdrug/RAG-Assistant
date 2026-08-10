"""Domain events for dynamic configuration changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class ConfigParameterChanged:
    """Published after a dynamic config parameter is committed to the database.

    Subscribers decide whether the changed key is relevant to them -- the
    event is broadcast generically and filtering happens inside each handler.
    """

    key: str
    old_value: str | None
    new_value: str
    value_type: str
    changed_by: int | None = None
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
