"""Conversation endpoints — thin wrappers around ConversationService."""

from __future__ import annotations

from application.services.conversation_service import ConversationService
from fastapi import APIRouter, Depends, Query

from presentation.api.auth_dependencies import get_current_user
from presentation.api.constants import CONFIDENCE_KEY, DEFAULT_PAGE_LIMIT, DEFAULT_PAGE_OFFSET
from presentation.api.dependencies import create_conversation_service
from presentation.api.helpers import filter_sources
from presentation.api.schemas import (
    ConversationHistoryResponse,
    ConversationListItem,
    ConversationListResponse,
    MessageResponse,
    NewConversationResponse,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    current_user: dict = Depends(get_current_user),
    service: ConversationService = Depends(create_conversation_service),
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=100),
    offset: int = Query(default=DEFAULT_PAGE_OFFSET, ge=0),
):
    items = await service.list_by_user(current_user["id"], limit=limit, offset=offset)
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
    current_user: dict = Depends(get_current_user),
    service: ConversationService = Depends(create_conversation_service),
):
    conv = await service.create(current_user["id"])
    return NewConversationResponse(conversation_id=conv.id)


@router.get("/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: int,
    current_user: dict = Depends(get_current_user),
    service: ConversationService = Depends(create_conversation_service),
):
    messages = await service.get_history(conversation_id, current_user["id"], current_user["role"])
    msg_responses = [
        MessageResponse(
            id=m.id,
            role=m.role,
            content=m.content,
            sources=filter_sources(m.sources, exclude_keys=frozenset({CONFIDENCE_KEY})),
            creation_date=m.creation_date,
        )
        for m in messages
    ]
    return ConversationHistoryResponse(conversation_id=conversation_id, messages=msg_responses)
