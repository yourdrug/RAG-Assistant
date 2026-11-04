"""Conversation endpoints — thin wrappers around ConversationService."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from application.chat import ConversationService
from presentation.api.auth_dependencies import get_current_user
from presentation.api.constants import CONFIDENCE_KEY, DEFAULT_PAGE_LIMIT, DEFAULT_PAGE_OFFSET
from presentation.api.dependencies import create_conversation_service
from presentation.api.helpers import filter_sources, persisted_id
from presentation.api.schemas import (
    ConversationHistoryResponse,
    ConversationListItem,
    ConversationListResponse,
    CurrentUser,
    MessageResponse,
    NewConversationResponse,
)


router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    *,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[ConversationService, Depends(create_conversation_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = DEFAULT_PAGE_LIMIT,
    offset: Annotated[int, Query(ge=0)] = DEFAULT_PAGE_OFFSET,
) -> ConversationListResponse:
    items = await service.list_by_user(current_user.id, limit=limit, offset=offset)
    return ConversationListResponse(
        conversations=[
            ConversationListItem(
                id=item.id,
                title=item.title,
                created_at=item.creation_date,
                message_count=item.message_count,
            )
            for item in items
        ]
    )


@router.post("", response_model=NewConversationResponse)
async def new_conversation(
    *,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[ConversationService, Depends(create_conversation_service)],
) -> NewConversationResponse:
    conv = await service.create(current_user.id)
    return NewConversationResponse(conversation_id=persisted_id(conv.id))


@router.get("/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    *,
    conversation_id: int,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    before_id: Annotated[int | None, Query(ge=1)] = None,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[ConversationService, Depends(create_conversation_service)],
) -> ConversationHistoryResponse:
    messages = await service.get_history(
        conversation_id, current_user.id, current_user.role, limit=limit + 1, before_id=before_id
    )
    has_more = len(messages) > limit
    messages = messages[-limit:]
    next_cursor = messages[0].id if has_more else None
    msg_responses = [
        MessageResponse(
            id=persisted_id(m.id),
            role=m.role,
            content=m.content,
            sources=filter_sources(m.sources, exclude_keys=frozenset({CONFIDENCE_KEY})),
            creation_date=m.creation_date,
        )
        for m in messages
    ]
    return ConversationHistoryResponse(
        conversation_id=conversation_id, messages=msg_responses, next_cursor=next_cursor
    )
