"""
api/routes/chat.py — Chat endpoints. Thin wrappers around ChatService.
"""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from infrastructure.auth import get_current_user
from infrastructure.database import get_db
from services.chat_service import ChatService
from sqlalchemy.orm import Session

from api.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat_stream(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    service = ChatService()

    async def event_generator():
        try:
            async for chunk in service.stream_chat(req.question, req.conversation_id, current_user, db):
                if chunk.startswith("\n__meta__:"):
                    meta = json.loads(chunk.replace("\n__meta__:", ""))
                    sources = meta.get("sources", [])
                    yield f"event: done\ndata: {json.dumps({'conversation_id': meta['conversation_id'], 'sources': sources}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    service = ChatService()
    answer, sources, conv_id = await service.sync_chat(req.question, req.conversation_id, current_user, db)
    return ChatResponse(answer=answer, conversation_id=conv_id, sources=sources)
