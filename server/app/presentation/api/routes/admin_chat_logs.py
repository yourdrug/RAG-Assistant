"""Admin chat logs endpoint — persistent Q&A quality tracking."""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from application.chat import ChatLogService
from application.identity import AssignmentService
from domain.chat.chat_log import ChatLog
from domain.chat.repositories import ChatLogFilter
from domain.identity.capabilities import Capability
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.rag.outcome import AnswerOutcome
from domain.shared.exceptions import EntityNotFoundError
from presentation.api.auth_dependencies import require_capability
from presentation.api.constants import DEFAULT_PAGE_LIMIT, DEFAULT_PAGE_OFFSET, MAX_PAGE_LIMIT
from presentation.api.dependencies import create_assignment_service, create_chat_log_service
from presentation.api.schemas import ChatLogEntry, ChatLogsResponse, CurrentUser


@dataclass(kw_only=True)
class ChatLogQuery:
    user_id: Annotated[int | None, Query()] = None
    domain: Annotated[str | None, Query()] = None
    space_id: Annotated[int | None, Query(gt=0)] = None
    outcome: Annotated[AnswerOutcome | None, Query()] = None
    model: Annotated[str | None, Query()] = None
    breadth: Annotated[Literal["narrow", "broad"] | None, Query()] = None
    config_hash: Annotated[str | None, Query(pattern="^[a-f0-9]{64}$")] = None
    sort_by: Annotated[Literal["date", "latency", "confidence"], Query()] = "date"
    sort_order: Annotated[Literal["asc", "desc"], Query()] = "desc"
    date_from: Annotated[datetime | None, Query()] = None
    date_to: Annotated[datetime | None, Query()] = None
    search: Annotated[str | None, Query()] = None
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = DEFAULT_PAGE_LIMIT
    offset: Annotated[int, Query(ge=0)] = DEFAULT_PAGE_OFFSET


router = APIRouter(tags=["admin-chat-logs"])


@router.get("/admin/chat-logs")
async def list_chat_logs(
    *,
    options: Annotated[ChatLogQuery, Depends()],
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.SYSTEM_OBSERVE))],
    service: Annotated[ChatLogService, Depends(create_chat_log_service)],
    assignment_service: Annotated[AssignmentService, Depends(create_assignment_service)],
) -> ChatLogsResponse:
    user_ids = await visible_user_ids(current_user, assignment_service)
    filters = ChatLogFilter(**asdict(options), user_ids=user_ids)
    total = await service.count_logs(filters)
    logs, email_map = await service.list_logs_with_emails(filters)
    entries = [
        entry_response(log, email_map.get(log.user_id) if log.user_id is not None else None)
        for log in logs
        if log.id is not None
    ]
    return ChatLogsResponse(logs=entries, total=total)


async def visible_user_ids(
    current_user: CurrentUser, assignment_service: AssignmentService
) -> tuple[int, ...] | None:
    if current_user.role != UserRole.CURATOR:
        return None

    scope = await assignment_service.get_scope(current_user.id)
    return tuple(scope["managed_client_ids"] + scope["managed_internal_ids"] + [current_user.id])


def entry_response(entry: ChatLog, email: str | None = None) -> ChatLogEntry:
    data = asdict(entry)
    data["creation_date"] = entry.creation_date.isoformat()
    return ChatLogEntry(**data, user_email=email)


@router.get("/admin/chat-logs/{log_id}")
async def get_chat_log(
    log_id: int,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.SYSTEM_OBSERVE))],
    service: Annotated[ChatLogService, Depends(create_chat_log_service)],
    assignment_service: Annotated[AssignmentService, Depends(create_assignment_service)],
) -> dict[str, Any]:
    result = await service.get_log(log_id, await visible_user_ids(current_user, assignment_service))

    if result is None:
        raise HTTPException(status_code=404, detail="Chat log not found")

    entry, snapshot = result
    return {"entry": entry_response(entry), "snapshot": service.compare_snapshot(snapshot)}


@router.get("/admin/rag-config-snapshots/{config_hash}")
async def get_config_snapshot(
    config_hash: str,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.SYSTEM_OBSERVE))],
    service: Annotated[ChatLogService, Depends(create_chat_log_service)],
    assignment_service: Annotated[AssignmentService, Depends(create_assignment_service)],
) -> dict[str, Any]:
    snapshot = await service.get_snapshot(
        config_hash, await visible_user_ids(current_user, assignment_service)
    )

    if snapshot is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")

    return service.compare_snapshot(snapshot)


class TraceChunkRequest(BaseModel):
    chunk_ids: list[Annotated[int, Field(gt=0)]] = Field(min_length=1, max_length=500)


@router.get("/admin/chat-logs/{log_id}/trace")
async def get_trace(
    log_id: int,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.SYSTEM_OBSERVE))],
    service: Annotated[ChatLogService, Depends(create_chat_log_service)],
    assignment_service: Annotated[AssignmentService, Depends(create_assignment_service)],
) -> dict[str, Any]:
    try:
        return await service.traces.get_trace(
            log_id, await visible_user_ids(current_user, assignment_service)
        )
    except EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Trace not found") from exc


@router.post("/admin/chat-logs/{log_id}/trace/chunks")
async def get_trace_chunks(
    log_id: int,
    body: TraceChunkRequest,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.SYSTEM_OBSERVE))],
    service: Annotated[ChatLogService, Depends(create_chat_log_service)],
    assignment_service: Annotated[AssignmentService, Depends(create_assignment_service)],
) -> dict[str, Any]:
    viewer = UserContext(
        user_id=current_user.id, user_kind=UserKind(current_user.kind), user_role=UserRole(current_user.role)
    )
    try:
        return await service.traces.read_chunks(
            log_id, body.chunk_ids, viewer, await visible_user_ids(current_user, assignment_service)
        )
    except EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Trace or chunk not found") from exc
