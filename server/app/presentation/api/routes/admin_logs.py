"""Admin logs endpoint — in-memory log viewer."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from application.admin import LogBufferPort
from presentation.api.auth_dependencies import require_admin
from presentation.api.constants import MAX_PAGE_LIMIT_LARGE
from presentation.api.dependencies import create_log_buffer
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
        )
        for entry in raw
    ]
    return LogsResponse(logs=logs, total=len(logs))
