"""Chat endpoints — thin wrappers around ChatService."""

from __future__ import annotations

import json
import uuid

from application.services.chat_service import ChatService
from domain.value_objects.stream_events import MetaEvent, StatusEvent, TextChunk
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from infrastructure.logging.actions import log_action
from shared import request_id_ctx

from presentation.api.auth_dependencies import get_current_user
from presentation.api.dependencies import create_chat_service
from presentation.api.rate_limits import chat_rate_limit
from presentation.api.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", dependencies=[Depends(chat_rate_limit)])
async def chat_stream(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
    chat_service: ChatService = Depends(create_chat_service),
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    req_id = uuid.uuid4().hex[:12]
    token = request_id_ctx.set(req_id)
    try:
        log_action(
            "chat",
            user_id=current_user["id"],
            details={
                "question": req.question[:100],
                "request_id": req_id,
            },
        )

        async def event_generator():
            try:
                yield ": heartbeat\n\n"
                async for event in chat_service.stream_chat(
                    req.question,
                    req.conversation_id,
                    current_user["id"],
                    current_user["kind"],
                    current_user["role"],
                    depth=req.depth,
                ):
                    if isinstance(event, MetaEvent):
                        sources = [s for s in event.sources if "_confidence" not in s]
                        payload = {
                            "conversation_id": event.conversation_id,
                            "sources": sources,
                            "confidence": event.confidence,
                            "request_id": req_id,
                        }
                        yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    elif isinstance(event, StatusEvent):
                        stage_payload = json.dumps({"stage": event.stage}, ensure_ascii=False)
                        yield f"event: status\ndata: {stage_payload}\n\n"
                    elif isinstance(event, TextChunk):
                        yield f"data: {json.dumps({'text': event.text}, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    finally:
        request_id_ctx.reset(token)


@router.post("/chat/sync", response_model=ChatResponse, dependencies=[Depends(chat_rate_limit)])
async def chat_sync(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
    chat_service: ChatService = Depends(create_chat_service),
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    req_id = uuid.uuid4().hex[:12]
    token = request_id_ctx.set(req_id)
    try:
        log_action(
            "chat.sync",
            user_id=current_user["id"],
            details={
                "question": req.question[:100],
                "request_id": req_id,
            },
        )

        result = await chat_service.sync_chat(
            req.question,
            req.conversation_id,
            current_user["id"],
            current_user["kind"],
            current_user["role"],
            depth=req.depth,
        )
        return ChatResponse(
            answer=result.answer,
            conversation_id=result.conversation_id,
            sources=result.sources,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
    finally:
        request_id_ctx.reset(token)
