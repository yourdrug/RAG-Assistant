"""UserRole and UserKind value objects -- role-based access control primitives."""

from __future__ import annotations

from enum import StrEnum

from domain.value_objects._base import ValidatedEnumMixin


class UserRole(ValidatedEnumMixin, StrEnum, label="role"):
    ADMIN = "admin"
    CURATOR = "curator"
    USER = "user"


class UserKind(ValidatedEnumMixin, StrEnum, label="kind"):
    INTERNAL = "internal"
    CLIENT = "client"
