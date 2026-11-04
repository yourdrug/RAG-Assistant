"""Exact substring search endpoint backed by PostgreSQL pg_trgm GIN index."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from application.ports.rate_limit import RateLimitPolicyName
from application.search import SearchService
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from presentation.api.auth_dependencies import get_current_user
from presentation.api.dependencies import create_search_service
from presentation.api.rate_limit import rate_limit
from presentation.api.schemas import CurrentUser, ExactSearchRequest, ExactSearchResponse, ExactSearchResult


logger = logging.getLogger("default")

router = APIRouter(tags=["search"])


@router.post(
    "/search/exact",
    response_model=ExactSearchResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.SEARCH))],
)
async def exact_search(
    *,
    req: ExactSearchRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    search_service: Annotated[SearchService, Depends(create_search_service)],
) -> ExactSearchResponse:
    """Exact substring search across all indexed chunks (Ctrl+F mode).

    Uses pg_trgm GIN index for fast ILIKE on millions of chunks.
    Requires min 3 characters for trigram index efficiency.
    """
    results, total = await search_service.exact_search(
        query=req.query,
        user=UserContext(
            user_id=current_user.id,
            user_kind=UserKind(current_user.kind),
            user_role=UserRole(current_user.role),
        ),
        limit=req.limit,
        offset=req.offset,
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
        total=total,
        limit=req.limit,
        offset=req.offset,
    )
