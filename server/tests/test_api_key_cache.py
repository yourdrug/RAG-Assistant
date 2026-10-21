"""API-key cache contracts through the authentication dependency."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from application.identity import ApiKeyService
from domain.shared.exceptions import AuthenticationError
from infrastructure.auth.api_key_provider import ApiKeyProvider
from presentation.api.auth_dependencies import authenticate_via_api_key


Cache = tuple[ApiKeyProvider, SimpleNamespace, dict[str, str]]


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> Cache:
    entries: dict[str, str] = {}

    async def get(key: str) -> str | None:
        return entries.get(key)

    async def setex(key: str, ttl: int, value: str) -> None:
        assert ttl == 30
        entries[key] = value

    redis = SimpleNamespace(get=AsyncMock(side_effect=get), setex=AsyncMock(side_effect=setex))
    monkeypatch.setattr(
        "infrastructure.auth.api_key_provider.redis_client", SimpleNamespace(async_redis=redis)
    )
    return ApiKeyProvider(), redis, entries


@pytest.mark.asyncio
async def test_invalid_key_is_looked_up_once_until_cache_expires(cache: Cache) -> None:
    provider, redis, entries = cache
    service = AsyncMock(spec=ApiKeyService)
    service.get_user_by_api_key_hash.return_value = None

    for _ in range(2):
        with pytest.raises(AuthenticationError, match="Invalid or revoked"):
            await authenticate_via_api_key("invalid-key", service, provider)

    service.get_user_by_api_key_hash.assert_awaited_once_with(provider.hash_key("invalid-key"))
    service.touch_api_key_last_used.assert_not_awaited()
    redis.setex.assert_awaited_once()

    entries.clear()  # Redis TTL expiry allows a fresh lookup.
    with pytest.raises(AuthenticationError):
        await authenticate_via_api_key("invalid-key", service, provider)
    assert service.get_user_by_api_key_hash.await_count == 2


@pytest.mark.asyncio
async def test_positive_cache_and_absent_entry_remain_distinct(cache: Cache) -> None:
    provider, _, _ = cache
    assert await provider.get_cached("absent") is provider.MISS
    value = {"api_key_id": 42, "id": 7, "is_active": True}
    await provider.set_cached("valid", value)
    assert await provider.get_cached("valid") == value


@pytest.mark.asyncio
async def test_redis_failure_falls_back_to_rejecting_database_lookup(cache: Cache) -> None:
    provider, redis, _ = cache
    redis.get.side_effect = OSError("Redis unavailable")
    service = AsyncMock(spec=ApiKeyService)
    service.get_user_by_api_key_hash.return_value = None
    with pytest.raises(AuthenticationError):
        await authenticate_via_api_key("invalid-key", service, provider)
    service.get_user_by_api_key_hash.assert_awaited_once()
