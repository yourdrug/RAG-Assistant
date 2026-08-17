"""Exact substring search endpoint backed by PostgreSQL pg_trgm GIN index."""

from __future__ import annotations

import logging

from application.services.search_service import SearchService
from fastapi import APIRouter, Depends

from presentation.api.auth_dependencies import get_current_user
from presentation.api.dependencies import get_search_service
from presentation.api.schemas import ExactSearchRequest, ExactSearchResponse, ExactSearchResult

logger = logging.getLogger("default")

router = APIRouter(tags=["search"])


@router.post("/search/exact", response_model=ExactSearchResponse)
async def exact_search(
    req: ExactSearchRequest,
    current_user: dict = Depends(get_current_user),
    search_service: SearchService = Depends(get_search_service),
):
    """Exact substring search across all indexed chunks (Ctrl+F mode).

    Uses pg_trgm GIN index for fast ILIKE on millions of chunks.
    Requires min 3 characters for trigram index efficiency.
    """
    results = await search_service.exact_search(
        query=req.query,
        user=current_user,
        limit=req.limit,
        mode=req.mode,
        document_id=req.document_id,
    )

    return ExactSearchResponse(
        query=req.query,
        results=[
            ExactSearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                filename=r.filename,
                content=r.content,
                chunk_index=r.chunk_index,
            )
            for r in results
        ],
        total=len(results),
    )
