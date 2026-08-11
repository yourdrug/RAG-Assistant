"""Semantic answer cache — stores and retrieves cached RAG answers.

Cache key is based on approximate embedding similarity (not exact text match)
combined with a visibility scope hash to prevent cross-tenant data leakage.
"""

from __future__ import annotations

import logging

from infrastructure.ml.hybrid import content_hash

log = logging.getLogger("default")


def compute_visibility_scope_hash(
    user_kind: str, group_ids: list[int], assigned_client_ids: list[int]
) -> str:
    """Deterministic hash of the user's ACL context.

    Two users with different group_ids or assigned_client_ids will produce
    different hashes, preventing cross-tenant cache hits.
    """
    scope = f"{user_kind}:{sorted(group_ids)}:{sorted(assigned_client_ids)}"
    return content_hash(scope)


def compute_question_hash(question_text: str) -> str:
    """Hash of the condensed question text for exact-match lookup."""
    return content_hash(question_text.strip().lower())


async def find_cached_answer(
    question_hash: str,
    visibility_scope_hash: str,
) -> dict | None:
    """Look up a cached answer by question hash + visibility scope.

    Returns the cache entry dict or None on miss.
    """
    from infrastructure.database.database import database

    try:
        async with database.master_node.async_session() as session:
            from infrastructure.database.models import AnswerCacheModel

            result = await session.execute(
                __import__("sqlalchemy")
                .select(AnswerCacheModel)
                .where(
                    AnswerCacheModel.question_embedding_hash == question_hash,
                    AnswerCacheModel.visibility_scope_hash == visibility_scope_hash,
                )
                .order_by(AnswerCacheModel.creation_date.desc())
                .limit(1)
            )
            entry = result.scalar_one_or_none()
            if entry is None:
                return None

            # Increment hit count
            entry.hit_count = (entry.hit_count or 0) + 1
            await session.commit()

            return {
                "answer": entry.answer,
                "sources": entry.sources or [],
                "question_text": entry.question_text,
                "hit_count": entry.hit_count,
            }
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
    from infrastructure.database.database import database

    try:
        async with database.master_node.async_session() as session:
            from infrastructure.database.models import AnswerCacheModel

            entry = AnswerCacheModel(
                question_text=question_text,
                question_embedding_hash=question_hash,
                answer=answer,
                sources=sources,
                visibility_scope_hash=visibility_scope_hash,
                document_ids=document_ids or [],
                hit_count=0,
            )
            session.add(entry)
            await session.commit()
            log.info("Cached answer for question hash=%s", question_hash[:12])
    except Exception:
        log.exception("Failed to store cached answer")


async def invalidate_cache_for_document(document_id: int) -> int:
    """Remove all cache entries that reference a specific document.

    Returns the number of deleted entries.
    """
    from infrastructure.database.database import database

    try:
        async with database.master_node.async_session() as session:
            from sqlalchemy import delete

            from infrastructure.database.models import AnswerCacheModel

            # Delete entries whose document_ids JSON array contains this document_id
            result = await session.execute(
                delete(AnswerCacheModel).where(AnswerCacheModel.document_ids.op("@>")(str([document_id])))
            )
            await session.commit()
            count = result.rowcount
            if count:
                log.info("Invalidated %d cache entries for document %d", count, document_id)
            return count
    except Exception:
        log.exception("Cache invalidation failed for document %d", document_id)
        return 0
