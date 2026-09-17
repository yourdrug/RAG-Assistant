"""User context — immutable snapshot of the authenticated user for access control."""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.value_objects.roles import UserKind, UserRole


@dataclass(frozen=True)
class UserContext:
    user_id: int
    user_kind: str
    user_role: str
    group_ids: tuple[int, ...] = field(default_factory=tuple)
    managed_client_ids: tuple[int, ...] = field(default_factory=tuple)
    managed_internal_ids: tuple[int, ...] = field(default_factory=tuple)
    managed_group_ids: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_ids", tuple(self.group_ids))
        object.__setattr__(self, "managed_client_ids", tuple(self.managed_client_ids))
        object.__setattr__(self, "managed_internal_ids", tuple(self.managed_internal_ids))
        object.__setattr__(self, "managed_group_ids", tuple(self.managed_group_ids))

    @property
    def is_client(self) -> bool:
        return self.user_kind == UserKind.CLIENT

    @property
    def is_admin(self) -> bool:
        return self.user_role == UserRole.ADMIN

    @property
    def is_curator(self) -> bool:
        return self.user_role == UserRole.CURATOR
