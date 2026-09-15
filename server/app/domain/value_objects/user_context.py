"""User context — immutable snapshot of the authenticated user for access control."""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.value_objects.roles import UserKind, UserRole


@dataclass(frozen=True)
class UserContext:
    user_id: int
    user_kind: str
    user_role: str
    group_ids: list[int] = field(default_factory=list)
    managed_client_ids: list[int] = field(default_factory=list)
    managed_internal_ids: list[int] = field(default_factory=list)
    managed_group_ids: list[int] = field(default_factory=list)

    @property
    def is_client(self) -> bool:
        return self.user_kind == UserKind.CLIENT

    @property
    def is_admin(self) -> bool:
        return self.user_role == UserRole.ADMIN

    @property
    def is_curator(self) -> bool:
        return self.user_role == UserRole.CURATOR
