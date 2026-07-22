"""DocumentVisibility Value Object."""

from __future__ import annotations

from enum import StrEnum

from domain.exceptions import ValidationError


class DocumentVisibility(StrEnum):
    INTERNAL_PUBLIC = "internal_public"
    INTERNAL_GROUP = "internal_group"
    INTERNAL_PRIVATE = "internal_private"
    CLIENT_PRIVATE = "client_private"

    @classmethod
    def validate(cls, value: str) -> DocumentVisibility:
        try:
            return cls(value)
        except ValueError:
            allowed = ", ".join(v.value for v in cls)
            raise ValidationError(f"visibility must be one of [{allowed}], got '{value}'") from None
