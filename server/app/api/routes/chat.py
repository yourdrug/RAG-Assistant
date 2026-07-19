import json

from config import settings
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from infrastructure.auth import get_current_user
from infrastructure.database import (
    get_db,
    get_history,
    get_or_create_conversation,
    save_message,
)
from infrastructure.vector_store import rag_invoke, rag_stream
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
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    conv_id = get_or_create_conversation(db, req.conversation_id, current_user["id"])
    save_message(db, conv_id, "user", req.question)

    history = get_history(db, conv_id, window=settings.history_window)
    if history and history[-1]["role"] == "user":
        history = history[:-1]

    async def event_generator():
        full_answer = ""
        sources = []

        try:
            async for chunk in rag_stream(req.question, history, current_user, db):
                if chunk.startswith("\n__sources__:"):
                    sources = json.loads(chunk.replace("\n__sources__:", ""))
                else:
                    full_answer += chunk
                    yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"

            save_message(db, conv_id, "assistant", full_answer, sources=sources)
            yield f"event: done\ndata: {json.dumps({'conversation_id': conv_id, 'sources': sources}, ensure_ascii=False)}\n\n"

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
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    conv_id = get_or_create_conversation(db, req.conversation_id, current_user["id"])
    save_message(db, conv_id, "user", req.question)

    history = get_history(db, conv_id, window=settings.history_window)
    if history and history[-1]["role"] == "user":
        history = history[:-1]

    answer, sources = await rag_invoke(req.question, history, current_user, db)
    save_message(db, conv_id, "assistant", answer, sources=sources)

    return ChatResponse(answer=answer, conversation_id=conv_id, sources=sources)
