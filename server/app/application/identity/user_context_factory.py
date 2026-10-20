"""UserContextFactory — builds UserContext from UnitOfWork (application-layer concern)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext


if TYPE_CHECKING:
    from application.uow import UnitOfWork


class UserContextFactory:
    """Constructs UserContext by loading group memberships and curator scope from DB.

    Extracted from the former UserContext.build() classmethod to keep the domain
    value object free of repository / uow dependencies.
    """

    async def build(
        self,
        uow: UnitOfWork,
        user_id: int,
        user_kind: str,
        user_role: str = UserRole.USER,
    ) -> UserContext:
        group_ids = (
            tuple(await uow.groups.get_user_group_ids(user_id)) if user_kind == UserKind.INTERNAL else ()
        )
        managed_client_ids: tuple[int, ...] = ()
        managed_internal_ids: tuple[int, ...] = ()
        managed_group_ids: tuple[int, ...] = ()

        if user_role == UserRole.CURATOR:
            scope = uow.assignments
            managed_client_ids = tuple(await scope.get_managed_client_ids(user_id))
            managed_internal_ids = tuple(await scope.get_managed_internal_ids(user_id))
            managed_group_ids = tuple(await scope.get_managed_group_ids(user_id))

        return UserContext(
            user_id=user_id,
            user_kind=UserKind(user_kind),
            user_role=UserRole(user_role),
            group_ids=group_ids,
            managed_client_ids=managed_client_ids,
            managed_internal_ids=managed_internal_ids,
            managed_group_ids=managed_group_ids,
        )
