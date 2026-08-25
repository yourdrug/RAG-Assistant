"""Admin logs endpoint — reads from in-memory buffer if available.

In Kubernetes, centralized logging via Loki/Grafana is the primary log
storage mechanism.  The buffer is empty unless attach_log_buffer() was
called at startup (it is not — kept for local dev/debug convenience).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from infrastructure.logging.log_buffer import log_buffer

from presentation.api.auth_dependencies import require_admin
from presentation.api.schemas import LogEntry, LogsResponse

router = APIRouter(tags=["admin-logs"])


@router.get("/admin/logs", response_model=LogsResponse)
async def list_logs(
    limit: int = Query(100, ge=1, le=500),
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
