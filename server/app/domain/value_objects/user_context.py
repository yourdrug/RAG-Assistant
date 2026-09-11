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
    managed_client_ids: list[int] = field(default_factory=list)
    managed_internal_ids: list[int] = field(default_factory=list)
    managed_group_ids: list[int] = field(default_factory=list)

    @classmethod
    async def build(cls, uow: Any, user_id: int, user_kind: str, user_role: str = "user") -> UserContext:
        """Construct a UserContext, loading group memberships for internal users.

        For CURATOR role, also loads managed scope from the curator scope repository.
        """
        group_ids = await uow.groups.get_user_group_ids(user_id) if user_kind == UserKind.INTERNAL else []
        managed_client_ids: list[int] = []
        managed_internal_ids: list[int] = []
        managed_group_ids: list[int] = []

        if user_role == UserRole.CURATOR:
            scope = uow.assignments
            managed_client_ids = await scope.get_managed_client_ids(user_id)
            managed_internal_ids = await scope.get_managed_internal_ids(user_id)
            managed_group_ids = await scope.get_managed_group_ids(user_id)

        return cls(
            user_id=user_id,
            user_kind=user_kind,
            user_role=user_role,
            group_ids=group_ids,
            managed_client_ids=managed_client_ids,
            managed_internal_ids=managed_internal_ids,
            managed_group_ids=managed_group_ids,
        )

    @property
    def is_client(self) -> bool:
        return self.user_kind == UserKind.CLIENT

    @property
    def is_admin(self) -> bool:
        return self.user_role == UserRole.ADMIN

    @property
    def is_curator(self) -> bool:
        return self.user_role == UserRole.CURATOR
