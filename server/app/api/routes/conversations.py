"""
api/routes/conversations.py — Conversation endpoints with Pydantic response models.
"""

from fastapi import APIRouter, Depends, HTTPException
from infrastructure.auth import get_current_user
from infrastructure.database import (
    create_conversation,
    get_conversation_owner,
    get_db,
    get_history,
)
from sqlalchemy.orm import Session

from api.schemas import ConversationHistoryResponse, MessageResponse, NewConversationResponse

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=NewConversationResponse)
async def new_conversation(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conv_id = create_conversation(db, current_user["id"])
    return NewConversationResponse(conversation_id=conv_id)


@router.get("/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    owner_id = get_conversation_owner(db, conversation_id)
    if owner_id is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if owner_id != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not your conversation")

    history = get_history(db, conversation_id, window=100)
    messages = [MessageResponse(role=m["role"], content=m["content"]) for m in history]
    return ConversationHistoryResponse(conversation_id=conversation_id, messages=messages)
