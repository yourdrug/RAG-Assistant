"""Application Service: AuthService — orchestrates auth use cases."""

from __future__ import annotations

from application.dto.auth_dto import CreateUserCommand, LoginCommand, LoginResult, UserDTO
from application.use_cases.auth.authenticate_user import AuthenticateUser
from application.use_cases.auth.create_user import CreateUser
from application.use_cases.auth.list_users import ListUsers
from application.use_cases.auth.toggle_user_active import ToggleUserActive


class AuthService:
    def __init__(
        self,
        authenticate_user: AuthenticateUser,
        create_user: CreateUser,
        list_users: ListUsers,
        toggle_user_active: ToggleUserActive,
    ) -> None:
        self._authenticate_user = authenticate_user
        self._create_user = create_user
        self._list_users = list_users
        self._toggle_user_active = toggle_user_active

    def authenticate(self, command: LoginCommand) -> LoginResult:
        return self._authenticate_user.execute(command)

    def create_user(self, command: CreateUserCommand, creator_role: str = "admin") -> UserDTO:
        return self._create_user.execute(command, creator_role)

    def list_users(self) -> list[UserDTO]:
        return self._list_users.execute()

    def toggle_active(self, user_id: int, is_active: bool, admin_id: int) -> dict:
        return self._toggle_user_active.execute(user_id, is_active, admin_id)
