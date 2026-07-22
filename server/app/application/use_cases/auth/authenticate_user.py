"""Use Case: AuthenticateUser — verify credentials and return JWT."""

from __future__ import annotations

from domain.exceptions import ValidationError
from domain.repositories.user_repository import UserRepository

from application.dto.auth_dto import LoginCommand, LoginResult


class AuthenticateUser:
    def __init__(
        self,
        user_repo: UserRepository,
        password_verifier,
        token_provider,
    ) -> None:
        self._user_repo = user_repo
        self._password_verifier = password_verifier
        self._token_provider = token_provider

    def execute(self, command: LoginCommand) -> LoginResult:
        user = self._user_repo.get_by_email(command.email)
        if user is None or not user.is_active:
            raise ValidationError("Invalid email or password")
        if not self._password_verifier.verify(command.password, user.hashed_password):
            raise ValidationError("Invalid email or password")

        token = self._token_provider.create_token(user_id=user.id, role=user.role)
        return LoginResult(access_token=token, role=user.role, kind=user.kind)
