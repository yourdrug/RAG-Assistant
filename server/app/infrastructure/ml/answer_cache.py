"""Semantic answer cache — Redis-backed.

Cache key is based on approximate embedding similarity (not exact text match)
combined with a visibility scope hash to prevent cross-tenant data leakage.
Uses Redis for fast in-memory lookups with TTL-based expiration.
"""

from __future__ import annotations

import gzip
import json
import logging
import time

from infrastructure.bm25.hybrid import content_hash
from infrastructure.redis.redis_client import redis_client
from domain.value_objects.curator_scope import CuratorScope
from domain.value_objects.roles import UserRole

log = logging.getLogger("default")

CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days
CACHE_PREFIX = "rag:cache:v3:"


def compute_visibility_scope_hash(
    user_kind: str,
    user_id: int,
    group_ids: list[int],
    user_role: str = UserRole.USER,
    curator_scope: CuratorScope | None = None,
) -> str:
    """Deterministic hash of the user's ACL context."""
    has_scope = curator_scope is not None and not curator_scope.is_empty()
    scope_repr = (
        f"curator:{sorted(curator_scope.managed_client_ids)}:"
        f"{sorted(curator_scope.managed_internal_ids)}:"
        f"{sorted(curator_scope.managed_group_ids)}"
        if has_scope
        else ""
    )
    return content_hash(f"{user_kind}:{user_role}:{user_id}:{sorted(group_ids)}:{scope_repr}")


def compute_question_hash(question_text: str) -> str:
    """Hash of the condensed question text for exact-match lookup."""
    return content_hash(question_text.strip().lower())


def _cache_key(question_hash: str, visibility_scope_hash: str) -> str:
    """Build Redis key for cache entry."""
    return f"{CACHE_PREFIX}{question_hash}:{visibility_scope_hash}"


async def find_cached_answer(
    question_hash: str,
    visibility_scope_hash: str,
    cache_enabled: bool = True,
) -> dict | None:
    """Look up a cached answer by question hash + visibility scope.

    Returns the cache entry dict or None on miss.
    """
    if not cache_enabled:
        return None

    try:
        r = redis_client.async_redis
        key = _cache_key(question_hash, visibility_scope_hash)
        raw = await r.get(key)
        if raw is None:
            return None

        entry = json.loads(gzip.decompress(raw))
        entry["hit_count"] = entry.get("hit_count", 0) + 1

        await r.set(key, gzip.compress(json.dumps(entry).encode()), keepttl=True)

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
    cache_enabled: bool = True,
) -> None:
    """Store a question-answer pair in the cache."""
    if not cache_enabled:
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
        await r.set(key, gzip.compress(json.dumps(entry).encode()), ex=CACHE_TTL_SECONDS)
        log.info("Cached answer for question hash=%s (ttl=%ds)", question_hash[:12], CACHE_TTL_SECONDS)
    except Exception:
        log.exception("Failed to store cached answer")


async def invalidate_by_document_ids(
    document_ids: list[int],
    cache_enabled: bool = True,
) -> int:
    """Invalidate all cache entries containing any of the given document_ids.

    Returns count of invalidated entries.
    """
    if not cache_enabled or not document_ids:
        return 0

    doc_id_set = set(document_ids)
    invalidated = 0

    try:
        r = redis_client.async_redis
        pattern = f"{CACHE_PREFIX}*"

        async for key in r.scan_iter(match=pattern, count=100):
            raw = await r.get(key)
            if raw is None:
                continue
            try:
                entry = json.loads(gzip.decompress(raw))
            except Exception:
                log.debug("Skipping corrupted cache entry %s", key)
                continue
            cached_doc_ids = entry.get("document_ids", [])
            if any(did in doc_id_set for did in cached_doc_ids):
                await r.delete(key)
                invalidated += 1

        if invalidated:
            log.info("Invalidated %d cache entries for document_ids=%s", invalidated, document_ids)
        return invalidated
    except Exception:
        log.exception("Cache invalidation failed")
        return 0
