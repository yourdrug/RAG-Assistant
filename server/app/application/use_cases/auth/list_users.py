"""Use Case: ListUsers — admin lists all users."""

from __future__ import annotations

from domain.repositories.user_repository import UserRepository

from application.dto.auth_dto import UserDTO


class ListUsers:
    def __init__(self, user_repo: UserRepository) -> None:
        self._user_repo = user_repo

    def execute(self) -> list[UserDTO]:
        users = self._user_repo.list_all()
        return [
            UserDTO(
                id=u.id,
                email=u.email,
                role=u.role,
                kind=u.kind,
                is_active=u.is_active,
            )
            for u in users
        ]
