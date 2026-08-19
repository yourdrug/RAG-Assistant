"""Application service for authentication and user management.

Provides login, registration, API-key CRUD, and token refresh.  Each public
method opens its own async UnitOfWork via the injected UnitOfWorkFactory,
keeping the service stateless and transaction-safe.
"""

from __future__ import annotations

from domain.entities.user import User
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from domain.services.password_hasher import IPasswordHasher
from domain.services.token_provider import ITokenProvider
from domain.value_objects.roles import UserKind, UserRole

from application.dto.auth_dto import CreateUserCommand, LoginCommand, LoginResult, UserDTO
from application.ports.api_key_provider import ApiKeyProviderPort
from application.ports.unit_of_work_factory import UnitOfWorkFactory


class AuthService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        password_hasher: IPasswordHasher,
        token_provider: ITokenProvider,
        api_key_provider: ApiKeyProviderPort,
    ) -> None:
        self._uow_factory = uow_factory
        self._hasher = password_hasher
        self._tokens = token_provider
        self._api_key_provider = api_key_provider

    async def authenticate(self, command: LoginCommand) -> LoginResult:
        async with self._uow_factory.create() as uow:
            user = await uow.users.get_by_email(command.email)
            if user is None or not user.is_active:
                raise ValidationError("Invalid email or password")
            if not self._hasher.verify(command.password, user.hashed_password):
                raise ValidationError("Invalid email or password")

            token = self._tokens.create_token(user_id=user.id, role=user.role)
            return LoginResult(access_token=token, role=user.role, kind=user.kind)

    async def create_user(
        self, command: CreateUserCommand, creator_role: str | UserRole = UserRole.ADMIN
    ) -> UserDTO:
        async with self._uow_factory.create(master=True) as uow:
            role = UserRole.validate(command.role)
            kind = UserKind.validate(command.kind)

            user = User(
                email=command.email,
                role=role,
                kind=kind,
            )
            user.ensure_valid_for_creation()
            user.can_be_created_by(UserRole(creator_role))

            if await uow.users.get_by_email(command.email) is not None:
                raise BusinessRuleViolation("User with this email already exists")

            user.hashed_password = self._hasher.hash(command.password)
            saved = await uow.users.save(user)

            return UserDTO(
                id=saved.id,
                email=saved.email,
                role=saved.role,
                kind=saved.kind,
                is_active=saved.is_active,
            )

    async def list_users(self) -> list[UserDTO]:
        async with self._uow_factory.create() as uow:
            users = await uow.users.list_all()
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

    async def toggle_active(self, user_id: int, is_active: bool, admin_id: int) -> dict:
        async with self._uow_factory.create(master=True) as uow:
            user = await uow.users.get_by_id(user_id)
            if user is None:
                raise EntityNotFound("User", user_id)
            user.deactivate_self_prohibited(admin_id)
            await uow.users.set_active(user_id, is_active)
            return {"id": user_id, "is_active": is_active}

    async def get_user_by_id(self, user_id: int) -> dict | None:
        """Return user as dict for auth lookups, or None if not found."""
        async with self._uow_factory.create() as uow:
            user = await uow.users.get_by_id(user_id)
            if user is None:
                return None
            return {
                "id": user.id,
                "email": user.email,
                "role": user.role,
                "kind": user.kind,
                "is_active": user.is_active,
            }

    async def get_user_by_api_key_hash(self, key_hash: str) -> dict | None:
        """Return user as dict for API key auth lookups, or None if not found."""
        async with self._uow_factory.create() as uow:
            result = await uow.api_keys.get_active_client_by_hash(key_hash)
            if result is None:
                return None
            return {
                "api_key_id": result.api_key_id,
                "id": result.id,
                "email": result.email,
                "role": result.role,
                "kind": result.kind,
                "is_active": result.is_active,
            }

    async def touch_api_key_last_used(self, api_key_id: int) -> None:
        """Update last_used_at for an API key."""
        async with self._uow_factory.create(master=True) as uow:
            await uow.api_keys.touch_last_used(api_key_id)

    def decode_token(self, token: str) -> dict:
        """Decode a JWT token.  Raises jwt exceptions on failure."""
        return self._tokens.decode_token(token)

    async def issue_api_key(self, client_user_id: int, name: str | None = None) -> dict:
        async with self._uow_factory.create(master=True) as uow:
            user = await uow.users.get_by_id(client_user_id)
            if user is None:
                raise EntityNotFound("User", client_user_id)
            if user.kind != UserKind.CLIENT:
                raise BusinessRuleViolation("API keys can only be issued to external (client) users")

            raw_key = self._api_key_provider.generate_key()
            key_hash = self._api_key_provider.hash_key(raw_key)
            prefix = self._api_key_provider.key_prefix_for_display(raw_key)

            saved = await uow.api_keys.create(
                user_id=client_user_id, key_hash=key_hash, key_prefix=prefix, name=name
            )

            return {
                "id": saved.id,
                "api_key": raw_key,
                "key_prefix": saved.key_prefix,
                "name": saved.name,
                "creation_date": saved.creation_date,
            }

    async def list_api_keys(self, client_user_id: int) -> list[dict]:
        async with self._uow_factory.create() as uow:
            keys = await uow.api_keys.list_for_user(client_user_id)
            return [
                {
                    "id": k.id,
                    "key_prefix": k.key_prefix,
                    "name": k.name,
                    "created_at": k.creation_date,
                    "revoked_at": k.revoked_at,
                    "last_used_at": k.last_used_at,
                    "is_active": k.is_active,
                }
                for k in keys
            ]

    async def revoke_api_key(self, api_key_id: int, client_user_id: int | None = None) -> dict:
        async with self._uow_factory.create(master=True) as uow:
            revoked = await uow.api_keys.revoke(api_key_id, user_id=client_user_id)
            if not revoked:
                raise EntityNotFound("ApiKey", api_key_id)

        await self._api_key_provider.invalidate_by_id(api_key_id)
        return {"id": api_key_id, "revoked": True}
