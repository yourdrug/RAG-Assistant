"""Immutable RAG options and the authenticated user's access snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain.identity.user_context import CuratorScope


if TYPE_CHECKING:
    from datetime import date

    from domain.identity.roles import UserKind, UserRole
    from domain.identity.user_context import UserContext


@dataclass(frozen=True)
class RagContext:
    user: UserContext
    depth: str | None = None
    summary: str | None = None
    as_of_date: date | None = None

    @property
    def user_id(self) -> int:
        return self.user.user_id

    @property
    def user_kind(self) -> UserKind:
        return self.user.user_kind

    @property
    def user_role(self) -> UserRole:
        return self.user.user_role

    @property
    def user_group_ids(self) -> tuple[int, ...]:
        return self.user.group_ids

    @property
    def curator_scope(self) -> CuratorScope | None:
        if not (
            self.user.managed_client_ids or self.user.managed_internal_ids or self.user.managed_group_ids
        ):
            return None

        return CuratorScope(
            self.user.managed_client_ids, self.user.managed_internal_ids, self.user.managed_group_ids
        )

    def to_user_context(self) -> UserContext:
        return self.user
