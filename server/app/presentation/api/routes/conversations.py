"""Conversation endpoints — thin wrappers around ConversationRepository."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from infrastructure.auth.fastapi_dependencies import get_current_user
from infrastructure.database.session import get_db
from sqlalchemy.orm import Session

from presentation.api.dependencies import get_repos
from presentation.api.schemas import ConversationHistoryResponse, MessageResponse, NewConversationResponse

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=NewConversationResponse)
async def new_conversation(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    repos = get_repos(db)
    conv = repos["conversation_repo"].create(current_user["id"])
    return NewConversationResponse(conversation_id=conv.id)


@router.get("/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    repos = get_repos(db)
    owner_id = repos["conversation_repo"].get_owner_id(conversation_id)
    if owner_id is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if owner_id != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not your conversation")

    messages = repos["message_repo"].get_history(conversation_id, window=100)
    msg_responses = [MessageResponse(role=m.role, content=m.content) for m in messages]
    return ConversationHistoryResponse(conversation_id=conversation_id, messages=msg_responses)
