"""Chat endpoints — thin wrappers around ChatService."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from infrastructure.auth.fastapi_dependencies import get_current_user
from infrastructure.database.session import get_db
from sqlalchemy.orm import Session

from presentation.api.dependencies import create_chat_service, get_repos
from presentation.api.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat_stream(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    repos = get_repos(db)
    group_ids = (
        repos["group_repo"].get_user_group_ids(current_user["id"])
        if current_user["kind"] == "internal"
        else []
    )
    assigned_ids = (
        repos["client_assignment_repo"].get_assigned_client_ids(current_user["id"])
        if current_user["kind"] == "internal"
        else []
    )

    service = create_chat_service(db)

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
    db: Session = Depends(get_db),
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    repos = get_repos(db)
    group_ids = (
        repos["group_repo"].get_user_group_ids(current_user["id"])
        if current_user["kind"] == "internal"
        else []
    )
    assigned_ids = (
        repos["client_assignment_repo"].get_assigned_client_ids(current_user["id"])
        if current_user["kind"] == "internal"
        else []
    )

    service = create_chat_service(db)
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
