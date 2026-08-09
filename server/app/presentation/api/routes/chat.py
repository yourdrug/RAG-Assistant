"""Chat endpoints — thin wrappers around ChatService."""

from __future__ import annotations

import json

from application.services.chat_service import ChatService
from domain.value_objects.stream_events import MetaEvent, TextChunk
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from infrastructure.logging.actions import log_action

from presentation.api.auth_dependencies import get_current_user
from presentation.api.dependencies import create_chat_service
from presentation.api.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat_stream(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
    chat_service: ChatService = Depends(create_chat_service),
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    log_action("chat", user_id=current_user["id"], details={"question": req.question[:100]})

    async def event_generator():
        try:
            async for event in chat_service.stream_chat(
                req.question,
                req.conversation_id,
                current_user["id"],
                current_user["kind"],
                current_user["role"],
                depth=req.depth,
            ):
                if isinstance(event, MetaEvent):
                    sources = event.sources
                    yield f"event: done\ndata: {json.dumps({'conversation_id': event.conversation_id, 'sources': sources}, ensure_ascii=False)}\n\n"
                elif isinstance(event, TextChunk):
                    yield f"data: {json.dumps({'text': event.text}, ensure_ascii=False)}\n\n"
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
    chat_service: ChatService = Depends(create_chat_service),
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    log_action("chat.sync", user_id=current_user["id"], details={"question": req.question[:100]})

    result = await chat_service.sync_chat(
        req.question,
        req.conversation_id,
        current_user["id"],
        current_user["kind"],
        current_user["role"],
        depth=req.depth,
    )
    return ChatResponse(answer=result.answer, conversation_id=result.conversation_id, sources=result.sources)
