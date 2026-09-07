"""BM25 index invalidation channel (C-5) — worker → API propagation via Redis pub/sub.

The BM25 index is per-process: the API loads it from S3 once and mutates it
only for in-process events, while the worker rebuilds/persists it. Without an
invalidation channel, documents ingested by the worker stay invisible to the
API's sparse (hybrid) search until the API restarts.

Contract: whenever a process persists a new BM25 index to S3, it publishes
``bm25:invalidate``; every other process that holds an in-memory copy
subscribes and reloads it lazily on next access (``MLClientRegistry.invalidate_bm25``).
"""

from __future__ import annotations

import asyncio
import logging

from infrastructure.persistence.redis_client import redis_client

log = logging.getLogger("default")

_CHANNEL = "bm25:invalidate"
_POLL_TIMEOUT_SEC = 5.0


async def publish_bm25_invalidation() -> None:
    """Notify other processes that the BM25 index was rebuilt/persisted.

    Best-effort: a lost message only delays the reload until the next save
    or the daily rebuild — never corrupts state.
    """
    try:
        r = redis_client.async_redis
        receivers = await r.publish(_CHANNEL, "invalidate")
        log.info("BM25 invalidation published (%d subscribers)", receivers)
    except Exception:
        log.warning("Failed to publish BM25 invalidation", exc_info=True)


async def listen_for_bm25_invalidation(ml_clients) -> None:
    """Subscribe loop: reload the in-memory BM25 index after external rebuilds.

    Runs as a background task in the API process. Cancelled on shutdown.
    """
    r = redis_client.async_redis
    pubsub = r.pubsub()
    await pubsub.subscribe(_CHANNEL)
    log.info("BM25 invalidation listener subscribed to '%s'", _CHANNEL)
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=_POLL_TIMEOUT_SEC)
            if message is None or message.get("type") != "message":
                continue
            if ml_clients is not None:
                ml_clients.invalidate_bm25()
                log.info("BM25 index invalidated — will reload from S3 on next access")
    except asyncio.CancelledError:
        raise
    finally:
        try:
            await pubsub.unsubscribe(_CHANNEL)
            await pubsub.aclose()
        except Exception:
            log.warning("Failed to close BM25 pubsub cleanly", exc_info=True)
