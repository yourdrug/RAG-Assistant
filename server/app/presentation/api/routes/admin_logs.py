"""Admin logs endpoint — in-memory log viewer."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from infrastructure.logging.log_buffer import log_buffer

from presentation.api.auth_dependencies import require_admin
from presentation.api.constants import MAX_PAGE_LIMIT_LARGE
from presentation.api.schemas import LogEntry, LogsResponse

router = APIRouter(tags=["admin-logs"])


@router.get("/admin/logs", response_model=LogsResponse)
async def list_logs(
    limit: int = Query(100, ge=1, le=MAX_PAGE_LIMIT_LARGE),
    level: str | None = Query(None),
    search: str | None = Query(None),
    admin: dict = Depends(require_admin),
):
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
