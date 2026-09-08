"""User context — immutable snapshot of the authenticated user for access control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.value_objects.roles import UserKind, UserRole


@dataclass(frozen=True)
class UserContext:
    user_id: int
    user_kind: str
    user_role: str
    group_ids: list[int] = field(default_factory=list)

    @classmethod
    async def build(cls, uow: Any, user_id: int, user_kind: str, user_role: str = "user") -> UserContext:
        """Construct a UserContext, loading group memberships for internal users."""
        group_ids = await uow.groups.get_user_group_ids(user_id) if user_kind == UserKind.INTERNAL else []
        return cls(user_id=user_id, user_kind=user_kind, user_role=user_role, group_ids=group_ids)

    @property
    def is_client(self) -> bool:
        return self.user_kind == UserKind.CLIENT

    @property
    def is_internal(self) -> bool:
        return self.user_kind == UserKind.INTERNAL

    @property
    def is_admin(self) -> bool:
        return self.user_role == UserRole.ADMIN
