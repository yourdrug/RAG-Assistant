"""Application Service: AuthService — manages authentication via UoWFactory.

Each method opens its own UnitOfWork. No db/session parameters.
"""

from __future__ import annotations

from domain.entities.user import User
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from domain.services.password_hasher import IPasswordHasher
from domain.services.token_provider import ITokenProvider
from domain.value_objects.roles import UserKind, UserRole
from infrastructure.uow_factory import UnitOfWorkFactory

from application.dto.auth_dto import CreateUserCommand, LoginCommand, LoginResult, UserDTO


class AuthService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        password_hasher: IPasswordHasher,
        token_provider: ITokenProvider,
    ) -> None:
        self._uow_factory = uow_factory
        self._hasher = password_hasher
        self._tokens = token_provider

    def authenticate(self, command: LoginCommand) -> LoginResult:
        with self._uow_factory.create() as uow:
            user = uow.users.get_by_email(command.email)
            if user is None or not user.is_active:
                raise ValidationError("Invalid email or password")
            if not self._hasher.verify(command.password, user.hashed_password):
                raise ValidationError("Invalid email or password")

            token = self._tokens.create_token(user_id=user.id, role=user.role)
            return LoginResult(access_token=token, role=user.role, kind=user.kind)

    def create_user(self, command: CreateUserCommand, creator_role: str = "admin") -> UserDTO:
        with self._uow_factory.create() as uow:
            role = UserRole.validate(command.role)
            kind = UserKind.validate(command.kind)

            user = User(
                email=command.email,
                role=role,
                kind=kind,
            )
            user.ensure_valid_for_creation()
            user.can_be_created_by(UserRole(creator_role))

            if uow.users.get_by_email(command.email) is not None:
                raise BusinessRuleViolation("User with this email already exists")

            user.hashed_password = self._hasher.hash(command.password)
            saved = uow.users.save(user)

            return UserDTO(
                id=saved.id,
                email=saved.email,
                role=saved.role,
                kind=saved.kind,
                is_active=saved.is_active,
            )

    def list_users(self) -> list[UserDTO]:
        with self._uow_factory.create() as uow:
            users = uow.users.list_all()
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

    def toggle_active(self, user_id: int, is_active: bool, admin_id: int) -> dict:
        with self._uow_factory.create() as uow:
            user = uow.users.get_by_id(user_id)
            if user is None:
                raise EntityNotFound("User", user_id)
            user.deactivate_self_prohibited(admin_id)
            uow.users.set_active(user_id, is_active)
            return {"id": user_id, "is_active": is_active}
