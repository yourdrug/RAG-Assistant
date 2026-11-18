"""Chat endpoints — thin wrappers around ChatService."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing
import json
import logging
from typing import Annotated
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from application.chat import ChatService, MessageFeedbackService
from application.ports.action_logger import ActionLoggerPort
from application.ports.rate_limit import RateLimitPolicyName
from application.rag.stream_events import MetaEvent, StatusEvent, StreamEvent, TextChunk
from domain.rag.exceptions import ContextBudgetExceededError
from domain.shared.exceptions import LLMUnavailableError, SemaphoreTimeoutError
from presentation.api.auth_dependencies import get_current_user
from presentation.api.constants import (
    CONFIDENCE_KEY,
    SSE_HEADERS,
    SSE_HEARTBEAT,
    SSE_MEDIA_TYPE,
)
from presentation.api.dependencies import (
    create_action_logger,
    create_chat_service,
    create_message_feedback_service,
)
from presentation.api.helpers import feedback_response, filter_sources
from presentation.api.rate_limit import rate_limit
from presentation.api.schemas import (
    ChatRequest,
    ChatResponse,
    CurrentUser,
    FeedbackRequest,
    FeedbackResponse,
)
from shared import request_id_ctx


router = APIRouter(tags=["chat"])

logger = logging.getLogger("default")


def _sse_error_event(code: str, message: str) -> str:
    """Build a uniform SSE error event payload."""
    return f"event: error\ndata: {json.dumps({'error': message, 'code': code}, ensure_ascii=False)}\n\n"


def _known_stream_error(exc: Exception) -> str:
    logger.warning("Chat stream unavailable (%s)", type(exc).__name__)

    if isinstance(exc, ContextBudgetExceededError):
        return _sse_error_event(exc.code, exc.message)

    if isinstance(exc, SemaphoreTimeoutError):
        return _sse_error_event("server_busy", "Сервер перегружен, повторите позже")

    return _sse_error_event("llm_unavailable", "LLM временно недоступен, попробуйте позже")


def format_sse_event(event: StreamEvent, req_id: str) -> str | None:
    """Format a single RAG pipeline event into an SSE data string."""
    if isinstance(event, MetaEvent):
        sources = filter_sources(event.sources, exclude_keys=frozenset({CONFIDENCE_KEY}))
        payload = {
            "conversation_id": event.conversation_id,
            "message_id": event.message_id,
            "sources": sources,
            "confidence": event.confidence,
            "as_of_date": event.as_of_date.isoformat() if event.as_of_date else None,
            "request_id": req_id,
            "answer": event.answer,
        }
        return f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    if isinstance(event, StatusEvent):
        stage_payload = json.dumps({"stage": event.stage}, ensure_ascii=False)
        return f"event: status\ndata: {stage_payload}\n\n"

    if isinstance(event, TextChunk):
        return f"data: {json.dumps({'text': event.text}, ensure_ascii=False)}\n\n"

    return None


@router.post(
    "/chat",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": (
                "Server-Sent Events (SSE), UTF-8. Frames end with a blank line; "
                "each data field contains JSON. Events: "
                '`message` (no event field): {"text": string}, an answer fragment; '
                '`status`: {"stage": string}, a progress update; '
                '`done`: {"conversation_id": integer, "message_id": integer|null, "sources": object[]|null, '
                '"confidence": number|null, "as_of_date": string|null, "request_id": string, '
                '"answer": string}, '
                "final metadata (as_of_date is an ISO date; answer is the authoritative final text, "
                "which replaces the streamed fragments); "
                '`error`: {"error": string, "code": string}, a terminal stream error. '
                "Errors after streaming starts retain HTTP 200. "
                "Comment frames (`: heartbeat`) carry no event data and should be ignored."
            ),
            "content": {
                SSE_MEDIA_TYPE: {
                    "schema": {"type": "string"},
                    "example": (
                        ": heartbeat\n\n"
                        'event: status\ndata: {"stage": "retrieval"}\n\n'
                        'data: {"text": "Hello"}\n\n'
                        'event: done\ndata: {"conversation_id": 1, "message_id": null, "sources": null, '
                        '"confidence": null, "as_of_date": null, "request_id": "abc123", "answer": ""}\n\n'
                    ),
                },
            },
        },
    },
    dependencies=[Depends(rate_limit(RateLimitPolicyName.CHAT))],
)
async def chat_stream(
    *,
    req: ChatRequest,
    request: Request,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    chat_service: Annotated[ChatService, Depends(create_chat_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> StreamingResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    req_id = uuid.uuid4().hex[:12]

    log(
        "chat",
        user_id=current_user.id,
        details={
            "question_chars": len(req.question),
            "request_id": req_id,
        },
    )

    return StreamingResponse(
        stream_chat_events(req, request, current_user, chat_service, req_id),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )


@router.post(
    "/chat/sync",
    response_model=ChatResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.CHAT))],
)
async def chat_sync(
    *,
    req: ChatRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    chat_service: Annotated[ChatService, Depends(create_chat_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    req_id = uuid.uuid4().hex[:12]
    token = request_id_ctx.set(req_id)
    try:
        log(
            "chat.sync",
            user_id=current_user.id,
            details={
                "question_chars": len(req.question),
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
            space_id=req.space_id,
        )
        return ChatResponse(
            answer=result.answer,
            conversation_id=result.conversation_id,
            message_id=result.message_id,
            confidence=result.confidence,
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


@router.put(
    "/chat/messages/{message_id}/feedback",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def rate_message(
    *,
    message_id: int,
    body: FeedbackRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[MessageFeedbackService, Depends(create_message_feedback_service)],
) -> FeedbackResponse:
    """Rate your own assistant answer; any other message is 404."""
    feedback = await service.rate(message_id, current_user.id, body.rating, body.reasons, body.comment)
    return feedback_response(feedback)


@router.delete(
    "/chat/messages/{message_id}/feedback",
    status_code=204,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def clear_message_feedback(
    *,
    message_id: int,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[MessageFeedbackService, Depends(create_message_feedback_service)],
) -> Response:
    await service.clear(message_id, current_user.id)
    return Response(status_code=204)


async def stream_chat_events(
    req: ChatRequest, request: Request, current_user: CurrentUser, chat_service: ChatService, req_id: str
) -> AsyncIterator[str]:
    token = request_id_ctx.set(req_id)
    try:
        yield SSE_HEARTBEAT

        async with aclosing(
            chat_service.stream_chat(
                req.question,
                req.conversation_id,
                current_user.id,
                current_user.kind,
                current_user.role,
                depth=req.depth,
                as_of_date=req.as_of_date,
                space_id=req.space_id,
            )
        ) as stream:
            async for event in stream:
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping stream (request_id=%s)", req_id)
                    break

                sse_data = format_sse_event(event, req_id)

                if sse_data is not None:
                    yield sse_data

    except (ContextBudgetExceededError, LLMUnavailableError, SemaphoreTimeoutError) as exc:
        yield _known_stream_error(exc)
    except TimeoutError:
        logger.warning("LLM auxiliary timeout", exc_info=True)
        yield _sse_error_event("llm_unavailable", "LLM временно недоступен, попробуйте позже")
    except Exception:
        logger.exception("Chat stream error")
        yield _sse_error_event("internal_error", "Internal error")
    finally:
        request_id_ctx.reset(token)
