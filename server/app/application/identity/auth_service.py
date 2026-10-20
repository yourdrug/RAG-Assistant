"""Password authentication, JWT decoding and current-user lookups."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from application.identity.dto import LoginResult, UserDTO
from domain.shared.exceptions import ValidationError


if TYPE_CHECKING:
    from application.identity.dto import LoginCommand
    from application.uow import UnitOfWorkFactory
    from domain.identity.ports import PasswordHasher, TokenProvider

# Fixed bcrypt hash used only to equalize login timing for unknown emails.
# The value is a valid bcrypt hash of an irrelevant string; its purpose is
# to force ~1 bcrypt round-trip so "user not found" ≈ "wrong password".
_DUMMY_BCRYPT_HASH = "$2b$12$C6UzMDM.H6dfI/f/IKcEcex8pZTQ4IcWm3qWD7.c9w4Z7l9m2oU2W"


class AuthService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        password_hasher: PasswordHasher,
        token_provider: TokenProvider,
    ) -> None:
        self._uow_factory = uow_factory
        self._hasher = password_hasher
        self._tokens = token_provider

    async def authenticate(self, command: LoginCommand) -> LoginResult:
        async with self._uow_factory.create(master=True) as uow:
            user = await uow.users.get_by_email(command.email)

            if user is None or not user.is_active:
                # Equalize timing: perform a bcrypt verify against a fixed
                # dummy hash so unknown emails are indistinguishable from
                # wrong-password attempts (avoids account enumeration).
                await self._hasher.verify(command.password, _DUMMY_BCRYPT_HASH)
                msg = "Invalid email or password"
                raise ValidationError(msg)

            if not await self._hasher.verify(command.password, user.hashed_password):
                msg = "Invalid email or password"
                raise ValidationError(msg)

            if user.id is None:
                msg = "User id is None for existing user"
                raise RuntimeError(msg)

            token = self._tokens.create_token(user_id=user.id, role=user.role)
            return LoginResult(access_token=token, role=user.role, kind=user.kind)

    async def get_user_by_id(self, user_id: int) -> UserDTO | None:
        """Return user as UserDTO for auth lookups, or None if not found."""
        async with self._uow_factory.create(master=True) as uow:
            user = await uow.users.get_by_id(user_id)

            if user is None:
                return None

            if user.id is None:
                msg = "User id is None for existing user"
                raise RuntimeError(msg)

            return UserDTO(
                id=user.id,
                email=user.email,
                role=user.role,
                kind=user.kind,
                is_active=user.is_active,
            )

    def decode_token(self, token: str) -> dict[str, Any]:
        """Decode a JWT token.  Raises jwt exceptions on failure."""
        return self._tokens.decode_token(token)
