"""User entity -- aggregate root for the Identity & Access bounded context.

Encapsulates credentials (password hash), role-based access control (UserRole,
UserKind), and group membership.  Enforces invariants such as role/kind
consistency on construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from domain.identity.roles import UserKind, UserRole
from domain.shared.exceptions import BusinessRuleViolationError, ValidationError


if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass
class User:
    id: int | None = None
    email: str = ""
    hashed_password: str = ""
    role: UserRole = UserRole.USER
    kind: UserKind = UserKind.INTERNAL
    is_active: bool = True
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))
    group_ids: list[int] = field(default_factory=list)
    # Support spaces a CLIENT may ask in; empty for internal users, who ask in any space.
    client_space_ids: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if isinstance(self.role, str):
            self.role = UserRole.parse(self.role)

        if isinstance(self.kind, str):
            self.kind = UserKind.parse(self.kind)

        self.client_space_ids = frozenset(self.client_space_ids)

    def ensure_can_be_created_by(self, creator_role: UserRole) -> None:
        """Enforce rule: only admin can create users."""
        if creator_role != UserRole.ADMIN:
            msg = "Only admin can create users"
            raise BusinessRuleViolationError(msg)

    def ensure_valid_for_creation(self) -> None:
        """Enforce rule: client cannot be admin or curator."""
        if self.kind == UserKind.CLIENT and self.role in (UserRole.ADMIN, UserRole.CURATOR):
            msg = "Client cannot be admin or curator"
            raise BusinessRuleViolationError(msg)

    def deactivate_self_prohibited(self, requester_id: int) -> None:
        """Enforce rule: admin cannot deactivate themselves."""
        if self.id == requester_id:
            msg = "Cannot deactivate yourself"
            raise BusinessRuleViolationError(msg)

    def assign_client_spaces(self, space_ids: Iterable[int]) -> None:
        if self.kind != UserKind.CLIENT:
            msg = "Only clients can be assigned a support space"
            raise BusinessRuleViolationError(msg)

        self.client_space_ids = frozenset(space_ids)

    def choose_client_space(self, requested: int | None) -> int:
        """The space a client asks in: the requested assigned one, or the only assigned one."""
        if not self.client_space_ids:
            msg = "No support space assigned to this client"
            raise ValidationError(msg, field="space_id")

        if requested is None:
            if len(self.client_space_ids) == 1:
                return next(iter(self.client_space_ids))

            msg = "space_id is required: choose one of your support spaces"
            raise ValidationError(msg, field="space_id")

        if requested not in self.client_space_ids:
            msg = "This knowledge space is not assigned to the client"
            raise ValidationError(msg, field="space_id")

        return requested

    def set_active(self, *, is_active: bool) -> None:
        self.is_active = is_active

    def change_role(self, new_role: UserRole) -> None:
        if self.kind == UserKind.CLIENT and new_role in (UserRole.ADMIN, UserRole.CURATOR):
            msg = "Client cannot be admin or curator"
            raise BusinessRuleViolationError(msg)

        self.role = new_role
