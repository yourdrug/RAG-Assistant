"""Use Case: ToggleUserActive — admin activates/deactivates a user."""

from __future__ import annotations

from domain.exceptions import EntityNotFound
from domain.repositories.user_repository import UserRepository


class ToggleUserActive:
    def __init__(self, user_repo: UserRepository) -> None:
        self._user_repo = user_repo

    def execute(self, user_id: int, is_active: bool, admin_id: int) -> dict:
        user = self._user_repo.get_by_id(user_id)
        if user is None:
            raise EntityNotFound("User", user_id)
        user.deactivate_self_prohibited(admin_id)
        self._user_repo.set_active(user_id, is_active)
        return {"id": user_id, "is_active": is_active}
