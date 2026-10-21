"""UserRole and UserKind value objects -- role-based access control primitives."""

from __future__ import annotations

from domain.shared.enums import ValidatedStrEnum


class UserRole(ValidatedStrEnum, label="role"):
    ADMIN = "admin"
    CURATOR = "curator"
    USER = "user"


class UserKind(ValidatedStrEnum, label="kind"):
    INTERNAL = "internal"
    CLIENT = "client"
