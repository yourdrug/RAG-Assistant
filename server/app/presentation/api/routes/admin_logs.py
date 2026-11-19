"""Admin logs endpoint — in-memory log viewer."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from application.admin import LogBufferPort
from application.identity import UserManagementService
from presentation.api.auth_dependencies import require_admin
from presentation.api.constants import MAX_PAGE_LIMIT_LARGE
from presentation.api.dependencies import create_log_buffer, create_user_management_service
from presentation.api.schemas import CurrentUser, LogEntry, LogsResponse


router = APIRouter(tags=["admin-logs"])


@router.get("/admin/logs", response_model=LogsResponse)
async def list_logs(
    *,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT_LARGE)] = 100,
    level: Annotated[str | None, Query()] = None,
    search: Annotated[str | None, Query()] = None,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    log_buffer: Annotated[LogBufferPort, Depends(create_log_buffer)],
    user_service: Annotated[UserManagementService, Depends(create_user_management_service)],
) -> LogsResponse:
    raw = log_buffer.get_logs(limit=limit, level=level, search=search)
    logs = [
        LogEntry(
            timestamp=entry["timestamp"],
            level=entry["level"],
            logger=entry["logger"],
            request_id=entry["request_id"],
            message=entry["message"],
            filename=entry.get("filename"),
            lineno=entry.get("lineno"),
            user_id=entry.get("user_id"),
        )
        for entry in raw
    ]
    user_ids = sorted({log.user_id for log in logs if log.user_id is not None})
    user_logins = await user_service.get_user_logins(user_ids)

    for log in logs:
        if log.user_id is not None:
            log.user_login = user_logins.get(log.user_id)

    return LogsResponse(logs=logs, total=len(logs))
