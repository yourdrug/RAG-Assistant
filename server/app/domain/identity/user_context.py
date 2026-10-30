"""User context — immutable snapshot of the authenticated user for access control."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from domain.identity.roles import UserKind, UserRole


@dataclass(frozen=True)
class UserContext:
    user_id: int
    user_kind: UserKind
    user_role: UserRole = UserRole.USER
    group_ids: tuple[int, ...] = field(default_factory=tuple)
    managed_client_ids: tuple[int, ...] = field(default_factory=tuple)
    managed_internal_ids: tuple[int, ...] = field(default_factory=tuple)
    managed_group_ids: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_kind", UserKind.parse(self.user_kind))
        object.__setattr__(self, "user_role", UserRole.parse(self.user_role))
        object.__setattr__(self, "group_ids", tuple(self.group_ids))
        object.__setattr__(self, "managed_client_ids", tuple(self.managed_client_ids))
        object.__setattr__(self, "managed_internal_ids", tuple(self.managed_internal_ids))
        object.__setattr__(self, "managed_group_ids", tuple(self.managed_group_ids))

    def with_curator_scope(self, scope: CuratorScope | None) -> UserContext:
        if scope is None:
            return self

        return replace(
            self,
            managed_client_ids=scope.managed_client_ids,
            managed_internal_ids=scope.managed_internal_ids,
            managed_group_ids=scope.managed_group_ids,
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


@dataclass(frozen=True)
class CuratorScope:
    """Snapshot of curator managed-scope at request time.

    None in RagContext means the user is not a curator (or has no managed IDs).
    Tuples (not lists) for immutability and hashability.
    """

    managed_client_ids: tuple[int, ...] = ()
    managed_internal_ids: tuple[int, ...] = ()
    managed_group_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "managed_client_ids", tuple(self.managed_client_ids))
        object.__setattr__(self, "managed_internal_ids", tuple(self.managed_internal_ids))
        object.__setattr__(self, "managed_group_ids", tuple(self.managed_group_ids))

    def is_empty(self) -> bool:
        return not (self.managed_client_ids or self.managed_internal_ids or self.managed_group_ids)
