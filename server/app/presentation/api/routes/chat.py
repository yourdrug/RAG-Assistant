"""Chat endpoints — thin wrappers around ChatService."""

from __future__ import annotations

import json
import logging
import uuid

from application.services.chat_service import ChatService
from domain.exceptions import LLMUnavailableError
from domain.value_objects.stream_events import MetaEvent, StatusEvent, TextChunk
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from shared import request_id_ctx

from presentation.api.auth_dependencies import get_current_user
from presentation.api.constants import (
    CONFIDENCE_KEY,
    QUESTION_LOG_MAX_CHARS,
    SSE_HEARTBEAT,
    SSE_HEADERS,
    SSE_MEDIA_TYPE,
)
from presentation.api.dependencies import create_action_logger, create_chat_service
from presentation.api.helpers import filter_sources
from presentation.api.schemas import ChatRequest, ChatResponse, CurrentUser

router = APIRouter(tags=["chat"])

logger = logging.getLogger("default")


def _sse_error_event(code: str, message: str) -> str:
    """Build a uniform SSE error event payload."""
    return f"event: error\ndata: {json.dumps({'error': message, 'code': code}, ensure_ascii=False)}\n\n"


def _format_sse_event(event, req_id: str) -> str | None:
    """Format a single RAG pipeline event into an SSE data string."""
    if isinstance(event, MetaEvent):
        sources = filter_sources(event.sources, exclude_keys=frozenset({CONFIDENCE_KEY}))
        payload = {
            "conversation_id": event.conversation_id,
            "sources": sources,
            "confidence": event.confidence,
            "request_id": req_id,
        }
        return f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
    if isinstance(event, StatusEvent):
        stage_payload = json.dumps({"stage": event.stage}, ensure_ascii=False)
        return f"event: status\ndata: {stage_payload}\n\n"
    if isinstance(event, TextChunk):
        return f"data: {json.dumps({'text': event.text}, ensure_ascii=False)}\n\n"
    return None


@router.post("/chat")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
    chat_service: ChatService = Depends(create_chat_service),
    log=Depends(create_action_logger),
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    req_id = uuid.uuid4().hex[:12]

    log(
        "chat",
        user_id=current_user.id,
        details={
            "question": req.question[:QUESTION_LOG_MAX_CHARS],
            "request_id": req_id,
        },
    )

    async def event_generator():
        token = request_id_ctx.set(req_id)
        try:
            yield SSE_HEARTBEAT
            async for event in chat_service.stream_chat(
                req.question,
                req.conversation_id,
                current_user.id,
                current_user.kind,
                current_user.role,
                depth=req.depth,
                as_of_date=req.as_of_date,
            ):
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping stream (request_id=%s)", req_id)
                    break
                sse_data = _format_sse_event(event, req_id)
                if sse_data is not None:
                    yield sse_data
        except LLMUnavailableError as exc:
            logger.warning("LLM unavailable (circuit breaker): %s", exc)
            yield _sse_error_event("llm_unavailable", "LLM временно недоступен, попробуйте позже")
        except TimeoutError:
            logger.warning("LLM auxiliary timeout", exc_info=True)
            yield _sse_error_event("llm_unavailable", "LLM временно недоступен, попробуйте позже")
        except Exception:
            logger.exception("Chat stream error")
            yield _sse_error_event("internal_error", "Internal error")
        finally:
            request_id_ctx.reset(token)

    return StreamingResponse(
        event_generator(),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )


@router.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(
    req: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
    chat_service: ChatService = Depends(create_chat_service),
    log=Depends(create_action_logger),
):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    req_id = uuid.uuid4().hex[:12]
    token = request_id_ctx.set(req_id)
    try:
        log(
            "chat.sync",
            user_id=current_user.id,
            details={
                "question": req.question[:QUESTION_LOG_MAX_CHARS],
                "request_id": req_id,
            },
        )

        result = await chat_service.sync_chat(
            req.question,
            req.conversation_id,
            current_user.id,
            current_user.kind,
            current_user.role,
            depth=req.depth,
            as_of_date=req.as_of_date,
        )
        return ChatResponse(
            answer=result.answer,
            conversation_id=result.conversation_id,
            sources=result.sources,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
    except LLMUnavailableError as exc:
        logger.warning("LLM unavailable (sync): %s", exc)
        raise HTTPException(
            status_code=503,
            detail="LLM временно недоступен, попробуйте позже",
            headers={"Retry-After": "30"},
        ) from None
    finally:
        request_id_ctx.reset(token)
