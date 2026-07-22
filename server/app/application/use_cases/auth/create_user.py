"""Use Case: CreateUser — admin creates a new user."""

from __future__ import annotations

from domain.entities.user import User
from domain.exceptions import BusinessRuleViolation
from domain.repositories.user_repository import UserRepository
from domain.value_objects.roles import UserKind, UserRole

from application.dto.auth_dto import CreateUserCommand, UserDTO


class CreateUser:
    def __init__(self, user_repo: UserRepository, password_hasher) -> None:
        self._user_repo = user_repo
        self._password_hasher = password_hasher

    def execute(self, command: CreateUserCommand, creator_role: str = "admin") -> UserDTO:
        role = UserRole.validate(command.role)
        kind = UserKind.validate(command.kind)

        user = User(
            email=command.email,
            role=role,
            kind=kind,
        )
        user.ensure_valid_for_creation()
        user.can_be_created_by(UserRole(creator_role))

        if self._user_repo.get_by_email(command.email) is not None:
            raise BusinessRuleViolation("User with this email already exists")

        user.hashed_password = self._password_hasher.hash(command.password)
        saved = self._user_repo.save(user)

        return UserDTO(
            id=saved.id,
            email=saved.email,
            role=saved.role,
            kind=saved.kind,
            is_active=saved.is_active,
        )
