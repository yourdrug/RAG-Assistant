"""Chat endpoints — thin wrappers around ChatService."""

from __future__ import annotations

import json

from application.services.chat_service import ChatService
from config import settings
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from infrastructure.auth.fastapi_dependencies import get_current_user
from infrastructure.ml.rag_service import RagService
from infrastructure.uow_factory import UnitOfWorkFactory

from presentation.api.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

_chat_service = ChatService(
    uow_factory=UnitOfWorkFactory(),
    rag_service=RagService(),
    history_window=settings.history_window,
)


@router.post("/chat")
async def chat_stream(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def event_generator():
        try:
            async for chunk in _chat_service.stream_chat(
                req.question,
                req.conversation_id,
                current_user["id"],
                current_user["kind"],
                current_user["role"],
            ):
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
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = await _chat_service.sync_chat(
        req.question,
        req.conversation_id,
        current_user["id"],
        current_user["kind"],
        current_user["role"],
    )
    return ChatResponse(
        answer=result.answer,
        conversation_id=result.conversation_id,
        sources=result.sources,
    )
