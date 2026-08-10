"""UserRole and UserKind value objects -- role-based access control primitives.

UserRole distinguishes admin vs regular users; UserKind distinguishes
internal (employee) vs external (client) accounts.  Both are validated
StrEnums that reject unknown values at construction time.
"""

from __future__ import annotations

from enum import StrEnum

from domain.exceptions import ValidationError


class UserRole(StrEnum):
    ADMIN = "admin"
    USER = "user"

    @classmethod
    def validate(cls, value: str) -> UserRole:
        try:
            return cls(value)
        except ValueError:
            raise ValidationError(f"role must be 'admin' or 'user', got '{value}'") from None


class UserKind(StrEnum):
    INTERNAL = "internal"
    CLIENT = "client"

    @classmethod
    def validate(cls, value: str) -> UserKind:
        try:
            return cls(value)
        except ValueError:
            raise ValidationError(f"kind must be 'internal' or 'client', got '{value}'") from None
