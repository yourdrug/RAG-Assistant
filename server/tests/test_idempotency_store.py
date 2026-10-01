"""Tests for Redis-backed write request reservations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fakeredis.aioredis
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from application.ports.idempotency_store import IdempotencyClaimStatus  # noqa: E402
from infrastructure.rate_limit.idempotency import IdempotencyStore  # noqa: E402


@pytest.mark.asyncio
async def test_claim_then_complete_can_be_replayed() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    store = IdempotencyStore(redis)
    try:
        reservation = await store.claim("key", "user:1", "upload", "fingerprint")
        assert reservation["status"] == IdempotencyClaimStatus.ACQUIRED.value

        await store.complete(reservation, {"document_id": 42})

        replay = await store.claim("key", "user:1", "upload", "fingerprint")
        assert replay == {
            "status": IdempotencyClaimStatus.COMPLETED.value,
            "result": {"document_id": 42},
        }
    finally:
        await redis.aclose()


@pytest.mark.asyncio
async def test_claim_reports_in_progress_and_payload_conflict() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    store = IdempotencyStore(redis)
    try:
        await store.claim("key", "user:1", "upload", "fingerprint")

        in_progress = await store.claim("key", "user:1", "upload", "fingerprint")
        conflict = await store.claim("key", "user:1", "upload", "different-fingerprint")

        assert in_progress["status"] == IdempotencyClaimStatus.IN_PROGRESS.value
        assert conflict["status"] == IdempotencyClaimStatus.CONFLICT.value
    finally:
        await redis.aclose()


@pytest.mark.asyncio
async def test_complete_rejects_a_reservation_owned_by_another_request() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    store = IdempotencyStore(redis)
    try:
        reservation = await store.claim("key", "user:1", "upload", "fingerprint")
        await redis.set(
            reservation["redis_key"],
            json.dumps({"state": IdempotencyClaimStatus.IN_PROGRESS.value, "token": "other-token"}),
        )

        with pytest.raises(RuntimeError, match="reservation was lost"):
            await store.complete(reservation, {"document_id": 42})
    finally:
        await redis.aclose()
