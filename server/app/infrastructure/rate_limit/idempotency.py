"""Atomic Redis reservations for idempotent write requests."""

from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any

from redis.exceptions import WatchError

from application.ports.idempotency_store import IdempotencyClaimStatus

IDEMPOTENCY_PREFIX = "rag:idempotency:v2:"
DEFAULT_TTL = 3600
_MAX_WATCH_RETRIES = 3


class IdempotencyStore:
    """Claim a request key once, then atomically attach its response."""

    def __init__(self, redis, ttl: int = DEFAULT_TTL) -> None:
        self._redis = redis
        self._ttl = ttl

    @staticmethod
    def _redis_key(key: str, principal_id: str) -> str:
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        principal_hash = hashlib.sha256(principal_id.encode("utf-8")).hexdigest()
        return f"{IDEMPOTENCY_PREFIX}{principal_hash}:{key_hash}"

    async def claim(
        self,
        key: str,
        principal_id: str,
        operation: str,
        fingerprint: str,
    ) -> dict[str, Any]:
        """Reserve an unused key or return its replay/conflict state."""
        redis_key = self._redis_key(key, principal_id)
        token = secrets.token_urlsafe(24)
        entry = {
            "operation": operation,
            "fingerprint": fingerprint,
            "state": IdempotencyClaimStatus.IN_PROGRESS.value,
            "token": token,
        }
        encoded = json.dumps(entry, separators=(",", ":"))
        if await self._redis.set(redis_key, encoded, nx=True, ex=self._ttl):
            return {
                "status": IdempotencyClaimStatus.ACQUIRED.value,
                "redis_key": redis_key,
                "token": token,
            }

        raw = await self._redis.get(redis_key)
        if raw is None:
            # The first reservation expired between SET NX and GET. Retry once
            # through the same atomic SET NX path.
            if await self._redis.set(redis_key, encoded, nx=True, ex=self._ttl):
                return {
                    "status": IdempotencyClaimStatus.ACQUIRED.value,
                    "redis_key": redis_key,
                    "token": token,
                }
            raw = await self._redis.get(redis_key)
        if raw is None:
            raise RuntimeError("Idempotency reservation disappeared during claim")

        existing = json.loads(raw)
        if existing.get("operation") != operation or existing.get("fingerprint") != fingerprint:
            return {"status": IdempotencyClaimStatus.CONFLICT.value}
        if existing.get("state") == IdempotencyClaimStatus.COMPLETED.value:
            return {
                "status": IdempotencyClaimStatus.COMPLETED.value,
                "result": existing.get("result"),
            }
        return {"status": IdempotencyClaimStatus.IN_PROGRESS.value}

    async def complete(self, reservation: dict[str, Any], result: dict[str, Any]) -> None:
        """Store the response only if this request still owns the reservation."""
        redis_key = reservation["redis_key"]
        token = reservation["token"]
        for _ in range(_MAX_WATCH_RETRIES):
            try:
                async with self._redis.pipeline(transaction=True) as pipe:
                    await pipe.watch(redis_key)
                    raw = await pipe.get(redis_key)
                    if raw is None:
                        raise RuntimeError("Idempotency reservation was lost before completion")
                    entry = json.loads(raw)
                    if not isinstance(entry, dict) or entry.get("token") != token:
                        raise RuntimeError("Idempotency reservation was lost before completion")

                    entry["state"] = IdempotencyClaimStatus.COMPLETED.value
                    entry["result"] = result
                    entry.pop("token", None)
                    pipe.multi()
                    pipe.set(
                        redis_key,
                        json.dumps(entry, default=str, separators=(",", ":")),
                        ex=self._ttl,
                    )
                    await pipe.execute()
                return
            except WatchError:
                continue
        raise RuntimeError("Idempotency reservation changed repeatedly before completion")
