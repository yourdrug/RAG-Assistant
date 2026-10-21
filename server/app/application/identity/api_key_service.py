"""API-key lifecycle and authentication lookups."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.identity.dto import ApiKeyAuthResult, ApiKeyInfo, IssueApiKeyResult
from domain.identity.roles import UserKind
from domain.shared.exceptions import BusinessRuleViolationError, EntityNotFoundError


if TYPE_CHECKING:
    from application.identity.ports import ApiKeyProviderPort
    from application.uow import UnitOfWorkFactory


class ApiKeyService:
    def __init__(self, uow_factory: UnitOfWorkFactory, api_key_provider: ApiKeyProviderPort) -> None:
        self._uow_factory = uow_factory
        self._api_key_provider = api_key_provider

    async def get_user_by_api_key_hash(self, key_hash: str) -> ApiKeyAuthResult | None:
        """Return user as ApiKeyAuthResult for API key auth lookups, or None if not found."""
        async with self._uow_factory.create(master=True) as uow:
            result = await uow.api_keys.get_active_client_by_hash(key_hash)

            if result is None:
                return None

            return ApiKeyAuthResult(
                api_key_id=result.api_key_id,
                id=result.id,
                email=result.email,
                role=result.role,
                kind=result.kind,
                is_active=result.is_active,
            )

    async def touch_api_key_last_used(self, api_key_id: int) -> None:
        """Update last_used_at for an API key."""
        async with self._uow_factory.create(master=True) as uow:
            await uow.api_keys.touch_last_used(api_key_id)

    async def issue_api_key(self, client_user_id: int, name: str | None = None) -> IssueApiKeyResult:
        async with self._uow_factory.create(master=True) as uow:
            user = await uow.users.get_by_id(client_user_id)

            if user is None:
                msg = "User"
                raise EntityNotFoundError(msg, client_user_id)

            if user.kind != UserKind.CLIENT:
                msg = "API keys can only be issued to external (client) users"
                raise BusinessRuleViolationError(msg)

            raw_key = self._api_key_provider.generate_key()
            key_hash = self._api_key_provider.hash_key(raw_key)
            prefix = self._api_key_provider.key_prefix_for_display(raw_key)

            saved = await uow.api_keys.create(
                user_id=client_user_id, key_hash=key_hash, key_prefix=prefix, name=name
            )

            if saved.id is None:
                msg = "ApiKey save returned None id"
                raise RuntimeError(msg)

            return IssueApiKeyResult(
                id=saved.id,
                api_key=raw_key,
                key_prefix=saved.key_prefix,
                name=saved.name,
                creation_date=saved.creation_date,
            )

    async def list_api_keys(self, client_user_id: int) -> list[ApiKeyInfo]:
        async with self._uow_factory.create() as uow:
            keys = await uow.api_keys.list_for_user(client_user_id)

            result = []

            for key in keys:
                if key.id is None:
                    msg = "ApiKey id is None for existing key"
                    raise RuntimeError(msg)

                result.append(
                    ApiKeyInfo(
                        id=key.id,
                        key_prefix=key.key_prefix,
                        name=key.name,
                        creation_date=key.creation_date,
                        revoked_at=key.revoked_at,
                        last_used_at=key.last_used_at,
                        is_active=key.is_active,
                    )
                )

            return result

    async def revoke_api_key(self, api_key_id: int, client_user_id: int | None = None) -> None:
        async with self._uow_factory.create(master=True) as uow:
            revoked = await uow.api_keys.revoke(api_key_id, user_id=client_user_id)

            if not revoked:
                msg = "ApiKey"
                raise EntityNotFoundError(msg, api_key_id)

        await self._api_key_provider.invalidate_by_id(api_key_id)
