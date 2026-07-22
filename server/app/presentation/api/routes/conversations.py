"""Conversation endpoints — thin wrappers around ConversationRepository."""

from __future__ import annotations

from application.uow import UnitOfWork
from fastapi import APIRouter, Depends, HTTPException
from infrastructure.auth.fastapi_dependencies import get_current_user

from presentation.api.dependencies import get_uow
from presentation.api.schemas import ConversationHistoryResponse, MessageResponse, NewConversationResponse

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=NewConversationResponse)
async def new_conversation(
    current_user: dict = Depends(get_current_user),
    uow: UnitOfWork = Depends(get_uow),
):
    conv = uow.conversations.create(current_user["id"])
    return NewConversationResponse(conversation_id=conv.id)


@router.get("/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: int,
    current_user: dict = Depends(get_current_user),
    uow: UnitOfWork = Depends(get_uow),
):
    owner_id = uow.conversations.get_owner_id(conversation_id)
    if owner_id is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if owner_id != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not your conversation")

    messages = uow.messages.get_history(conversation_id, window=100)
    msg_responses = [MessageResponse(role=m.role, content=m.content) for m in messages]
    return ConversationHistoryResponse(conversation_id=conversation_id, messages=msg_responses)
