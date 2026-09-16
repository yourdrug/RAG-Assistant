"""Auth & user schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from domain.value_objects.roles import UserKind, UserRole
from presentation.api.constants import AUTH_SCHEME_BEARER


class CurrentUser(BaseModel):
    """Authenticated user context returned by auth dependencies."""

    model_config = ConfigDict(extra="forbid")

    id: int
    email: str
    role: str
    kind: str
    is_active: bool
    api_key_id: int | None = None


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class CreateUserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: UserRole | None = None
    kind: UserKind | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = AUTH_SCHEME_BEARER


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    kind: str
    is_active: bool


class MeResponse(BaseModel):
    """Response for GET /auth/me — explicit schema without api_key_id."""

    model_config = ConfigDict(extra="forbid")

    id: int
    email: str
    role: str
    kind: str
    is_active: bool


class UserListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    users: list[UserResponse]


class ChangeRoleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: UserRole


class CuratorScopeResponse(BaseModel):
    managed_client_ids: list[int]
    managed_internal_ids: list[int]
    managed_group_ids: list[int]
