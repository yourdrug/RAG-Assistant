"""Admin chat logs endpoint — persistent Q&A quality tracking."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from infrastructure.uow_factory import UnitOfWorkFactory

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import get_uow_factory
from presentation.api.schemas import ChatLogEntry, ChatLogsResponse

router = APIRouter(tags=["admin-chat-logs"])


@router.get("/admin/chat-logs", response_model=ChatLogsResponse)
async def list_chat_logs(
    user_id: int | None = Query(None),
    domain: str | None = Query(None),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    search: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    admin: dict = Depends(require_admin),
    uow_factory: UnitOfWorkFactory = Depends(get_uow_factory),
):
    async with uow_factory.create() as uow:
        total = await uow.chat_logs.count_logs(
            user_id=user_id,
            domain=domain,
            date_from=date_from,
            date_to=date_to,
            search=search,
        )
        logs = await uow.chat_logs.list_logs(
            user_id=user_id,
            domain=domain,
            date_from=date_from,
            date_to=date_to,
            search=search,
            limit=limit,
            offset=offset,
        )

    entries = [
        ChatLogEntry(
            id=log.id,
            creation_date=log.creation_date.isoformat() if log.creation_date else "",
            user_id=log.user_id,
            conversation_id=log.conversation_id,
            question=log.question,
            answer=log.answer,
            sources=log.sources,
            latency_ms=log.latency_ms,
            model_used=log.model_used,
            breadth=log.breadth,
            domain=log.domain,
            retrieval_count=log.retrieval_count,
            reranker_score=log.reranker_score,
        )
        for log in logs
    ]
    return ChatLogsResponse(logs=entries, total=total)
