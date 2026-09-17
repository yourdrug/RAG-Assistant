"""Semantic answer cache — Redis-backed.

Cache key is based on approximate embedding similarity (not exact text match)
combined with a visibility scope hash to prevent cross-tenant data leakage.
Uses Redis for fast in-memory lookups with TTL-based expiration.

Invalidation uses a reverse index (rag:doc:{doc_id} → Set[cache_key]) for
O(1) lookup instead of O(N) SCAN.  Legacy entries created before the reverse
index are handled via graceful fallback: if the set is empty, we scan only
the keys matching that specific document.
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
DOC_INDEX_PREFIX = "rag:doc:"
DOC_INDEX_TTL = CACHE_TTL_SECONDS


def compute_visibility_scope_hash(
    user_kind: str,
    user_id: int,
    group_ids: list[int],
    user_role: str = UserRole.USER,
    curator_scope: CuratorScope | None = None,
) -> str:
    """Deterministic hash of the user's ACL context."""
    if curator_scope is not None and not curator_scope.is_empty():
        scope_repr = (
            f"curator:{sorted(curator_scope.managed_client_ids)}:"
            f"{sorted(curator_scope.managed_internal_ids)}:"
            f"{sorted(curator_scope.managed_group_ids)}"
        )
    else:
        scope_repr = ""
    return content_hash(f"{user_kind}:{user_role}:{user_id}:{sorted(group_ids)}:{scope_repr}")


def compute_question_hash(question_text: str) -> str:
    """Hash of the condensed question text for exact-match lookup."""
    return content_hash(question_text.strip().lower())


def _cache_key(question_hash: str, visibility_scope_hash: str) -> str:
    """Build Redis key for cache entry."""
    return f"{CACHE_PREFIX}{question_hash}:{visibility_scope_hash}"


def _doc_index_key(doc_id: int) -> str:
    """Build Redis key for reverse index set."""
    return f"{DOC_INDEX_PREFIX}{doc_id}"


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

        if document_ids:
            pipe = r.pipeline()
            for doc_id in document_ids:
                pipe.sadd(_doc_index_key(doc_id), key)
                pipe.expire(_doc_index_key(doc_id), DOC_INDEX_TTL)
            await pipe.execute()

        log.info("Cached answer for question hash=%s (ttl=%ds)", question_hash[:12], CACHE_TTL_SECONDS)
    except Exception:
        log.exception("Failed to store cached answer")


async def _delete_cache_entries(keys: list[str]) -> int:
    """Delete cache entries and clean up their reverse index links atomically."""
    if not keys:
        return 0
    r = redis_client.async_redis
    pipe = r.pipeline()
    for key in keys:
        pipe.get(key)
    raw_results = await pipe.execute()

    pipe2 = r.pipeline()
    for key, raw in zip(keys, raw_results, strict=True):
        if raw is not None:
            try:
                entry = json.loads(gzip.decompress(raw))
                for doc_id in entry.get("document_ids", []):
                    pipe2.srem(_doc_index_key(doc_id), key)
            except Exception:
                log.debug("Failed to clean reverse index for %s", key)
        pipe2.delete(key)
    await pipe2.execute()
    return sum(1 for raw in raw_results if raw is not None)


async def _legacy_scan_for_doc_ids(document_ids: list[int], skip_keys: set[str]) -> set[str]:
    """Fallback: scan all cache keys to find entries matching document_ids.

    Used only when reverse index sets are missing (legacy entries).
    """
    r = redis_client.async_redis
    found: set[str] = set()
    doc_id_set = set(document_ids)
    pattern = f"{CACHE_PREFIX}*"
    async for key in r.scan_iter(match=pattern, count=100):
        if key in skip_keys:
            continue
        raw = await r.get(key)
        if raw is None:
            continue
        try:
            entry = json.loads(gzip.decompress(raw))
        except Exception:
            log.debug("Skipping corrupted cache entry %s", key)
            continue
        if any(did in doc_id_set for did in entry.get("document_ids", [])):
            found.add(key)
    return found


async def invalidate_by_document_ids(
    document_ids: list[int],
    cache_enabled: bool = True,
) -> int:
    """Invalidate all cache entries containing any of the given document_ids.

    Returns count of invalidated entries.
    """
    if not cache_enabled or not document_ids:
        return 0

    try:
        r = redis_client.async_redis
        keys_to_delete: set[str] = set()
        legacy_scan_needed = False

        for doc_id in document_ids:
            members = await r.smembers(_doc_index_key(doc_id))  # type: ignore[misc]
            if members:
                keys_to_delete.update(m for m in members)
            else:
                legacy_scan_needed = True

        if legacy_scan_needed:
            keys_to_delete |= await _legacy_scan_for_doc_ids(document_ids, keys_to_delete)

        invalidated = await _delete_cache_entries(list(keys_to_delete))

        if invalidated:
            log.info("Invalidated %d cache entries for document_ids=%s", invalidated, document_ids)
        return invalidated
    except Exception:
        log.exception("Cache invalidation failed")
        return 0
