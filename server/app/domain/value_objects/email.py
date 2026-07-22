"""Email Value Object — immutable, validated, compared by value."""

from __future__ import annotations

import re
from dataclasses import dataclass

from domain.exceptions import ValidationError

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


@dataclass(frozen=True)
class Email:
    value: str

    def __post_init__(self) -> None:
        normalized = self.value.strip().lower()
        if not normalized:
            raise ValidationError("Email cannot be empty")
        if not _EMAIL_RE.match(normalized):
            raise ValidationError(f"Invalid email format: {self.value}")
        object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:
        return self.value
