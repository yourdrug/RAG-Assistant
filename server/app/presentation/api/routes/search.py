"""Search endpoints — exact substring search via pg_trgm."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from presentation.api.auth_dependencies import get_current_user
from presentation.api.dependencies import get_uow_factory
from presentation.api.schemas import ExactSearchRequest, ExactSearchResponse, ExactSearchResult

logger = logging.getLogger("default")

router = APIRouter(tags=["search"])


@router.post("/search/exact", response_model=ExactSearchResponse)
async def exact_search(
    req: ExactSearchRequest,
    current_user: dict = Depends(get_current_user),
):
    """Exact substring search across all indexed chunks (Ctrl+F mode).

    Uses pg_trgm GIN index for fast ILIKE on millions of chunks.
    Requires min 3 characters for trigram index efficiency.
    """
    from infrastructure.repositories.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository

    uow_factory = get_uow_factory()

    # Fetch ACL data for the current user
    async with uow_factory.create() as uow:
        group_ids = await uow.groups.get_user_group_ids(current_user["id"])
        assigned_client_ids = (
            await uow.client_assignments.get_assigned_client_ids(current_user["id"])
            if current_user["kind"] == "internal"
            else []
        )

        repo = SQLAlchemyChunkRepository(uow.session)
        results = await repo.search_substring(
            query=req.query,
            user=current_user,
            group_ids=group_ids,
            assigned_client_ids=assigned_client_ids,
            limit=req.limit,
            mode=req.mode,
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
