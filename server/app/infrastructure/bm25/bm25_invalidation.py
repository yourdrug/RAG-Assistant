"""Optional invalidation channel for standalone local BM25 snapshot tools.

Production Qdrant sparse search does not subscribe to this channel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from infrastructure.redis.redis_client import redis_client


if TYPE_CHECKING:
    from infrastructure.bm25.snapshots import LocalBM25Snapshots

log = logging.getLogger("default")

_CHANNEL = "bm25:invalidate"
_POLL_TIMEOUT_SEC = 5.0


async def publish_bm25_invalidation(*, raise_on_error: bool = False) -> None:
    """Notify other processes that the BM25 index was rebuilt/persisted.

    By default failures are logged. Durable outbox callers request propagation
    so a failed publication is retried.
    """
    try:
        r = redis_client.async_redis
        receivers = await r.publish(_CHANNEL, "invalidate")
        log.info("BM25 invalidation published (%d subscribers)", receivers)
    except Exception:
        log.warning("Failed to publish BM25 invalidation", exc_info=True)

        if raise_on_error:
            raise


async def listen_for_bm25_invalidation(snapshots: LocalBM25Snapshots) -> None:
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

            if snapshots is not None:
                snapshots.invalidate_bm25()
                log.info("BM25 index invalidated — will reload from S3 on next access")
    finally:
        try:
            await pubsub.unsubscribe(_CHANNEL)
            await pubsub.close()
        except Exception:
            log.warning("Failed to close BM25 pubsub cleanly", exc_info=True)
