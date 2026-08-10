"""Static API key provider -- generation, hashing, and in-memory cache verification.

Static API keys are an authentication method for external (kind='client')
users only. Unlike JWT, the key is verified against the database (via a
short-TTL cache) and can be revoked instantly without cryptography -- similar
to Stripe/OpenAI key schemes.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from threading import Lock

_KEY_PREFIX = "rg_sys_"
_CACHE_TTL_SECONDS = 30.0  # намеренно короткий TTL: отзыв ключа должен применяться быстро


@dataclass
class _CacheEntry:
    value: dict | None
    expires_at: float


class ApiKeyProvider:
    """Кэширует sha256(ключ) -> данные пользователя на несколько секунд,
    чтобы не бить в Postgres на каждый запрос от одного и того же клиента.

    invalidate_by_id() сбрасывает запись немедленно при отзыве ключа (в рамках
    одного инстанса сервера); при нескольких инстансах отзыв применится у
    остальных не позже, чем через _CACHE_TTL_SECONDS — это компромисс
    in-memory кэша без внешнего Redis.
    """

    MISS = object()

    def __init__(self, ttl_seconds: float = _CACHE_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, _CacheEntry] = {}
        self._hash_by_id: dict[int, str] = {}
        self._lock = Lock()

    @staticmethod
    def generate_key() -> str:
        """Сгенерировать новый ключ в открытом виде. Показывается пользователю ОДИН РАЗ."""
        return f"{_KEY_PREFIX}{secrets.token_urlsafe(32)}"

    @staticmethod
    def hash_key(raw_key: str) -> str:
        """Детерминированный sha256-хеш для хранения/поиска.

        В отличие от паролей, API-ключи уже высокоэнтропийны, поэтому быстрый
        детерминированный хеш (а не bcrypt) — стандартная практика (так делают
        GitHub, Stripe): он позволяет искать по индексу в БД.
        """
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @staticmethod
    def key_prefix_for_display(raw_key: str) -> str:
        return raw_key[: len(_KEY_PREFIX) + 6]

    def get_cached(self, key_hash: str):
        with self._lock:
            entry = self._cache.get(key_hash)
            if entry is None or entry.expires_at < time.monotonic():
                self._cache.pop(key_hash, None)
                return self.MISS
            return entry.value

    def set_cached(self, key_hash: str, value: dict | None) -> None:
        with self._lock:
            self._cache[key_hash] = _CacheEntry(value=value, expires_at=time.monotonic() + self._ttl)
            if value is not None:
                self._hash_by_id[value["api_key_id"]] = key_hash

    def invalidate_by_id(self, api_key_id: int) -> None:
        with self._lock:
            key_hash = self._hash_by_id.pop(api_key_id, None)
            if key_hash is not None:
                self._cache.pop(key_hash, None)


api_key_provider = ApiKeyProvider()
