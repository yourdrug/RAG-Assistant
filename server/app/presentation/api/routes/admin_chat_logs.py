"""Admin chat logs endpoint — persistent Q&A quality tracking."""

from __future__ import annotations

from datetime import datetime

from application.services.chat_log_service import ChatLogService
from application.services.assignment_service import AssignmentService
from domain.value_objects.capabilities import Capability
from domain.value_objects.roles import UserRole
from fastapi import APIRouter, Depends, Query

from presentation.api.auth_dependencies import require_capability
from presentation.api.constants import DEFAULT_PAGE_LIMIT, DEFAULT_PAGE_OFFSET, MAX_PAGE_LIMIT
from presentation.api.dependencies import create_chat_log_service, create_assignment_service
from presentation.api.schemas import ChatLogEntry, ChatLogsResponse, CurrentUser

router = APIRouter(tags=["admin-chat-logs"])


@router.get("/admin/chat-logs", response_model=ChatLogsResponse)
async def list_chat_logs(
    user_id: int | None = Query(None),
    domain: str | None = Query(None),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    search: str | None = Query(None),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(DEFAULT_PAGE_OFFSET, ge=0),
    current_user: CurrentUser = Depends(require_capability(Capability.SYSTEM_OBSERVE)),
    service: ChatLogService = Depends(create_chat_log_service),
    assignment_service: AssignmentService = Depends(create_assignment_service),
):
    # Curator sees only logs of managed users (+ self); admin sees all
    user_ids_filter = None
    if current_user.role == UserRole.CURATOR:
        scope = await assignment_service.get_scope(current_user.id)
        managed_ids = scope["managed_client_ids"] + scope["managed_internal_ids"] + [current_user.id]
        user_ids_filter = managed_ids

    total = await service.count_logs(
        user_id=user_id,
        user_ids=user_ids_filter,
        domain=domain,
        date_from=date_from,
        date_to=date_to,
        search=search,
    )
    logs, email_map = await service.list_logs_with_emails(
        user_id=user_id,
        user_ids=user_ids_filter,
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
            user_email=email_map.get(log.user_id) if log.user_id else None,
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
            input_tokens=log.input_tokens,
            output_tokens=log.output_tokens,
        )
        for log in logs
        if log.id is not None
    ]
    return ChatLogsResponse(logs=entries, total=total)
