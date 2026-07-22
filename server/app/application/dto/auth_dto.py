"""Auth-related DTOs."""

from __future__ import annotations

from dataclasses import dataclass


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
    role: str = "user"
    kind: str = "internal"


@dataclass(frozen=True)
class UserDTO:
    id: int
    email: str
    role: str
    kind: str
    is_active: bool
