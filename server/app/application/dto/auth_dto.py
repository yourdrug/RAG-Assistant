"""Auth-related DTOs -- immutable data-transfer objects for login, registration, and user info."""

from __future__ import annotations

from dataclasses import dataclass

from domain.value_objects.roles import UserKind, UserRole


@dataclass(frozen=True)
class LoginCommand:
    email: str
    password: str


@dataclass(frozen=True)
class LoginResult:
    access_token: str
    role: str
    kind: str


@dataclass(frozen=True)
class CreateUserCommand:
    email: str
    password: str
    role: str = UserRole.USER
    kind: str = UserKind.INTERNAL


@dataclass(frozen=True)
class UserDTO:
    id: int
    email: str
    role: str
    kind: str
    is_active: bool
