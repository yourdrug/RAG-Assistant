"""HTTP helpers for request-scoped idempotency reservations."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from fastapi import HTTPException

from application.ports.idempotency_store import IdempotencyClaimStatus, IdempotencyStorePort


def request_fingerprint(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


async def reserve_idempotency(
    store: IdempotencyStorePort,
    key: str | None,
    principal_id: str,
    operation: str,
    payload: object,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return a cached result or a reservation owned by this request."""
    if key is None:
        return None, None

    claim = await store.claim(key, principal_id, operation, request_fingerprint(payload))
    status = claim.get("status")

    if status == IdempotencyClaimStatus.COMPLETED:
        result = claim.get("result")

        if not isinstance(result, dict):
            message = "Completed idempotency entry has no stored response"
            raise RuntimeError(message)

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

    message = "Idempotency store returned an invalid claim state"
    raise RuntimeError(message)


async def complete_idempotency(
    store: IdempotencyStorePort, reservation: dict[str, Any] | None, response: dict[str, Any]
) -> None:
    if reservation is not None:
        await store.complete(reservation, response)
