"""HTTP helpers for request-scoped idempotency reservations."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import HTTPException

from application.ports.idempotency_store import IdempotencyClaimStatus


def request_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


async def reserve_idempotency(
    store,
    key: str | None,
    principal_id: str,
    operation: str,
    payload: Any,
) -> tuple[dict | None, dict | None]:
    """Return a cached result or a reservation owned by this request."""
    if key is None:
        return None, None
    claim = await store.claim(key, principal_id, operation, request_fingerprint(payload))
    status = claim.get("status")
    if status == IdempotencyClaimStatus.COMPLETED:
        result = claim.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Completed idempotency entry has no stored response")
        return result, None
    if status == IdempotencyClaimStatus.ACQUIRED:
        return None, claim
    if status == IdempotencyClaimStatus.IN_PROGRESS:
        raise HTTPException(status_code=409, detail="A request with this Idempotency-Key is still running")
    if status == IdempotencyClaimStatus.CONFLICT:
        raise HTTPException(
            status_code=409,
            detail="Idempotency-Key was already used for a different operation or request body",
        )
    raise RuntimeError("Idempotency store returned an invalid claim state")


async def complete_idempotency(store, reservation: dict | None, response: dict) -> None:
    if reservation is not None:
        await store.complete(reservation, response)
