"""Chat endpoints — thin wrappers around ChatService."""

from __future__ import annotations

import json

from application.services.chat_service import ChatService
from application.uow import UnitOfWork
from application.use_cases.chat.stream_chat import StreamChat
from application.use_cases.chat.sync_chat import SyncChat
from config import settings
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from infrastructure.auth.fastapi_dependencies import get_current_user
from infrastructure.ml.rag_service import RagService

from presentation.api.dependencies import get_uow
from presentation.api.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

_rag_service = RagService()


def _chat_service(uow: UnitOfWork) -> ChatService:
    return ChatService(
        stream_chat=StreamChat(
            conversation_repo=uow.conversations,
            message_repo=uow.messages,
            rag_service=_rag_service,
            settings=settings,
        ),
        sync_chat=SyncChat(
            conversation_repo=uow.conversations,
            message_repo=uow.messages,
            rag_service=_rag_service,
            settings=settings,
        ),
    )


@router.post("/chat")
async def chat_stream(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
    uow: UnitOfWork = Depends(get_uow),
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    group_ids = (
        uow.groups.get_user_group_ids(current_user["id"]) if current_user["kind"] == "internal" else []
    )
    assigned_ids = (
        uow.client_assignments.get_assigned_client_ids(current_user["id"])
        if current_user["kind"] == "internal"
        else []
    )

    service = _chat_service(uow)

    async def event_generator():
        try:
            async for chunk in service.stream_chat(
                req.question,
                req.conversation_id,
                current_user["id"],
                current_user["kind"],
                current_user["role"],
                group_ids,
                assigned_ids,
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
    uow: UnitOfWork = Depends(get_uow),
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    group_ids = (
        uow.groups.get_user_group_ids(current_user["id"]) if current_user["kind"] == "internal" else []
    )
    assigned_ids = (
        uow.client_assignments.get_assigned_client_ids(current_user["id"])
        if current_user["kind"] == "internal"
        else []
    )

    service = _chat_service(uow)
    result = await service.sync_chat(
        req.question,
        req.conversation_id,
        current_user["id"],
        current_user["kind"],
        current_user["role"],
        group_ids,
        assigned_ids,
    )
    return ChatResponse(
        answer=result.answer,
        conversation_id=result.conversation_id,
        sources=result.sources,
    )
