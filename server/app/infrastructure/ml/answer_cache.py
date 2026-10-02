"""Exact-match answer cache — Redis-backed.

Cache keys include normalized question text, request context and a visibility
scope hash to prevent cross-tenant data leakage.
Uses Redis for fast in-memory lookups with TTL-based expiration.

Invalidation uses a reverse index (rag:doc:{doc_id} → Set[cache_key]) for
O(1) lookup instead of O(N) SCAN. Entries without an index expire by TTL.
"""

from __future__ import annotations

import base64
import gzip
import json
import logging
import time
from collections.abc import Awaitable
from typing import cast

from domain.utils import content_hash
from infrastructure.redis.redis_client import redis_client
from domain.value_objects.curator_scope import CuratorScope
from domain.value_objects.roles import UserRole

log = logging.getLogger("default")

CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days
CACHE_PREFIX = "rag:cache:v4:"
CORPUS_REVISION_KEY = "rag:corpus:revision"
DOC_INDEX_PREFIX = "rag:doc:"
DOC_INDEX_TTL = CACHE_TTL_SECONDS

# The revision check and write must be one Redis operation: a generation
# started before a mutation must not repopulate an invalidated answer.
_STORE_IF_CURRENT = """
if (redis.call('GET', KEYS[1]) or '0') ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[2], ARGV[2], 'EX', ARGV[3])
for i = 3, #KEYS do
    redis.call('SADD', KEYS[i], KEYS[2])
    redis.call('EXPIRE', KEYS[i], ARGV[3])
end
return 1
"""


async def get_corpus_revision() -> str | None:
    """Disable cache for this request if its consistency fence is unavailable."""
    try:
        return str(await redis_client.async_redis.get(CORPUS_REVISION_KEY) or "0")
    except Exception:
        log.exception("Cache revision lookup failed")
        return None


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


def compute_question_hash(question_text: str, context: dict | None = None) -> str:
    """Hash question and answer-affecting request context for exact-match lookup."""
    normalized = question_text.strip().lower()
    if context:
        normalized = json.dumps(
            {"question": normalized, "context": context},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    return content_hash(normalized)


def _cache_key(question_hash: str, visibility_scope_hash: str) -> str:
    """Build Redis key for cache entry."""
    return f"{CACHE_PREFIX}{question_hash}:{visibility_scope_hash}"


def _doc_index_key(doc_id: int) -> str:
    """Build Redis key for reverse index set."""
    return f"{DOC_INDEX_PREFIX}{doc_id}"


def _encode_entry(entry: dict) -> str:
    # RedisClient uses decode_responses=True for the shared connection.
    return base64.b64encode(gzip.compress(json.dumps(entry).encode())).decode("ascii")


def _decode_entry(raw) -> dict:
    if isinstance(raw, bytes) and raw.startswith(b"\x1f\x8b"):
        return json.loads(gzip.decompress(raw))
    return json.loads(gzip.decompress(base64.b64decode(raw, validate=True)))


async def find_cached_answer(
    question_hash: str,
    visibility_scope_hash: str,
    cache_enabled: bool = True,
    expected_revision: str | None = None,
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

        entry = _decode_entry(raw)
        if expected_revision is not None and await get_corpus_revision() != expected_revision:
            return None
        entry["hit_count"] = entry.get("hit_count", 0) + 1

        await r.set(key, _encode_entry(entry), keepttl=True, xx=True)

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
    expected_revision: str | None = None,
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
        if expected_revision is not None:
            indexes = [_doc_index_key(doc_id) for doc_id in set(document_ids or [])]
            # Redis shares the EVAL stub between sync and async clients; this
            # connection is always redis.asyncio.Redis, so the result is awaitable.
            await cast(
                Awaitable[object],
                r.eval(
                    _STORE_IF_CURRENT,
                    2 + len(indexes),
                    CORPUS_REVISION_KEY,
                    key,
                    *indexes,
                    expected_revision,
                    _encode_entry(entry),
                    str(CACHE_TTL_SECONDS),
                ),
            )
            return
        pipe = r.pipeline(transaction=True)
        pipe.set(key, _encode_entry(entry), ex=CACHE_TTL_SECONDS)
        if document_ids:
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
                entry = _decode_entry(raw)
                for doc_id in entry.get("document_ids", []):
                    pipe2.srem(_doc_index_key(doc_id), key)
            except Exception:
                log.debug("Failed to clean reverse index for %s", key)
        pipe2.delete(key)
    await pipe2.execute()
    return sum(1 for raw in raw_results if raw is not None)


async def invalidate_by_document_ids(
    document_ids: list[int],
    cache_enabled: bool = True,
    raise_on_error: bool = False,
) -> int:
    """Invalidate all cache entries containing any of the given document_ids.

    Returns count of invalidated entries. Durable outbox callers set
    ``raise_on_error`` to retry failures; already committed HTTP mutations
    use best-effort immediate invalidation and leave retries to the outbox.
    """
    # Mutations must invalidate old entries even while cache reads are disabled.
    if not document_ids:
        return 0

    try:
        r = redis_client.async_redis
        await r.incr(CORPUS_REVISION_KEY)
        keys_to_delete: set[str] = set()

        for doc_id in document_ids:
            members = await r.smembers(_doc_index_key(doc_id))  # type: ignore[misc]
            if members:
                keys_to_delete.update(m for m in members)

        invalidated = await _delete_cache_entries(list(keys_to_delete))

        if invalidated:
            log.info("Invalidated %d cache entries for document_ids=%s", invalidated, document_ids)
        return invalidated
    except Exception:
        log.exception("Cache invalidation failed")
        if raise_on_error:
            raise
        return 0
