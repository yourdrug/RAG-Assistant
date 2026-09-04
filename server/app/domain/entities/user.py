"""User entity -- aggregate root for the Identity & Access bounded context.

Encapsulates credentials (password hash), role-based access control (UserRole,
UserKind), and group membership.  Enforces invariants such as role/kind
consistency on construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from domain.exceptions import BusinessRuleViolation
from domain.value_objects.roles import UserKind, UserRole


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

    def __post_init__(self) -> None:
        if isinstance(self.role, str):
            self.role = UserRole.validate(self.role)
        if isinstance(self.kind, str):
            self.kind = UserKind.validate(self.kind)

    def ensure_can_be_created_by(self, creator_role: UserRole) -> None:
        """Enforce rule: only admin can create users."""
        if creator_role != UserRole.ADMIN:
            raise BusinessRuleViolation("Only admin can create users")

    def ensure_valid_for_creation(self) -> None:
        """Enforce rule: client cannot be admin."""
        if self.kind == UserKind.CLIENT and self.role == UserRole.ADMIN:
            raise BusinessRuleViolation("Client cannot be admin")

    def deactivate_self_prohibited(self, requester_id: int) -> None:
        """Enforce rule: admin cannot deactivate themselves."""
        if self.id == requester_id:
            raise BusinessRuleViolation("Cannot deactivate yourself")

    def toggle_active(self, is_active: bool) -> None:
        self.is_active = is_active

    def change_role(self, new_role: UserRole) -> None:
        if self.kind == UserKind.CLIENT and new_role == UserRole.ADMIN:
            raise BusinessRuleViolation("Client cannot be admin")
        self.role = new_role
