"""Semantic answer cache — Redis-backed.

Cache key is based on approximate embedding similarity (not exact text match)
combined with a visibility scope hash to prevent cross-tenant data leakage.
Uses Redis for fast in-memory lookups with TTL-based expiration.
"""

from __future__ import annotations

import json
import logging
import time

from config import settings

from infrastructure.ml.hybrid import content_hash
from infrastructure.persistence.redis_client import redis_client

log = logging.getLogger("default")

CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days
CACHE_PREFIX = "rag:cache:"


def compute_visibility_scope_hash(user_kind: str, group_ids: list[int]) -> str:
    """Deterministic hash of the user's ACL context.

    Two users with different group_ids will produce different hashes,
    preventing cross-tenant cache hits.
    """
    scope = f"{user_kind}:{sorted(group_ids)}"
    return content_hash(scope)


def compute_question_hash(question_text: str) -> str:
    """Hash of the condensed question text for exact-match lookup."""
    return content_hash(question_text.strip().lower())


def _cache_key(question_hash: str, visibility_scope_hash: str) -> str:
    """Build Redis key for cache entry."""
    return f"{CACHE_PREFIX}{question_hash}:{visibility_scope_hash}"


async def find_cached_answer(
    question_hash: str,
    visibility_scope_hash: str,
) -> dict | None:
    """Look up a cached answer by question hash + visibility scope.

    Returns the cache entry dict or None on miss.
    """
    if not settings.cache_enabled:
        return None

    try:
        r = redis_client.async_redis
        key = _cache_key(question_hash, visibility_scope_hash)
        raw = await r.get(key)
        if raw is None:
            return None

        entry = json.loads(raw)
        entry["hit_count"] = entry.get("hit_count", 0) + 1

        # Update hit count (don't extend TTL on hit)
        await r.set(key, json.dumps(entry), ex=CACHE_TTL_SECONDS)

        return entry
    except Exception:
        log.exception("Cache lookup failed")
        return None


async def store_cached_answer(
    question_text: str,
    question_hash: str,
    answer: str,
    sources: list[dict],
    visibility_scope_hash: str,
    document_ids: list[int] | None = None,
) -> None:
    """Store a question-answer pair in the cache."""
    if not settings.cache_enabled:
        return

    try:
        r = redis_client.async_redis
        key = _cache_key(question_hash, visibility_scope_hash)
        entry = {
            "question_text": question_text,
            "answer": answer,
            "sources": sources,
            "document_ids": document_ids or [],
            "hit_count": 0,
            "created_at": time.time(),
        }
        await r.set(key, json.dumps(entry), ex=CACHE_TTL_SECONDS)
        log.info("Cached answer for question hash=%s (ttl=%ds)", question_hash[:12], CACHE_TTL_SECONDS)
    except Exception:
        log.exception("Failed to store cached answer")
