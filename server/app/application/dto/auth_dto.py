"""Auth-related DTOs -- immutable data-transfer objects for login, registration, and user info."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

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


@dataclass(frozen=True)
class ToggleActiveResult:
    id: int
    is_active: bool


@dataclass(frozen=True)
class ChangeRoleCommand:
    user_id: int
    new_role: str


@dataclass(frozen=True)
class ApiKeyAuthResult:
    api_key_id: int
    id: int
    email: str
    role: str
    kind: str
    is_active: bool


@dataclass(frozen=True)
class IssueApiKeyResult:
    id: int
    api_key: str
    key_prefix: str
    name: str | None
    creation_date: datetime | None


@dataclass(frozen=True)
class ApiKeyInfo:
    id: int
    key_prefix: str
    name: str | None
    creation_date: datetime | None
    revoked_at: datetime | None
    last_used_at: datetime | None
    is_active: bool
