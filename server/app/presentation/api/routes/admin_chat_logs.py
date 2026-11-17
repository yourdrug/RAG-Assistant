"""Admin chat logs endpoint — persistent Q&A quality tracking."""

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import io
import json
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from application.chat import ChatLogRow, ChatLogService
from application.chat.review import BenchmarkTransfer
from application.identity import AssignmentService
from application.ports.rate_limit import RateLimitPolicyName
from domain.chat.feedback import FeedbackReason
from domain.chat.repositories import ChatLogFilter
from domain.chat.review import ChatLogReview, ReviewStatus
from domain.identity.capabilities import Capability
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.rag.outcome import AnswerOutcome
from domain.shared.exceptions import EntityNotFoundError
from presentation.api.auth_dependencies import require_capability
from presentation.api.constants import DEFAULT_PAGE_LIMIT, DEFAULT_PAGE_OFFSET, MAX_PAGE_LIMIT
from presentation.api.dependencies import create_assignment_service, create_chat_log_service
from presentation.api.helpers import feedback_response
from presentation.api.rate_limit import rate_limit
from presentation.api.schemas import (
    BenchmarkDraftResponse,
    BenchmarkTransferRequest,
    ChatLogEntry,
    ChatLogReviewRequest,
    ChatLogReviewResponse,
    ChatLogsResponse,
    CurrentUser,
)


@dataclass(kw_only=True)
class ChatLogQuery:
    user_id: Annotated[int | None, Query()] = None
    domain: Annotated[str | None, Query()] = None
    space_id: Annotated[int | None, Query(gt=0)] = None
    outcome: Annotated[AnswerOutcome | None, Query()] = None
    model: Annotated[str | None, Query()] = None
    breadth: Annotated[Literal["narrow", "broad"] | None, Query()] = None
    config_hash: Annotated[str | None, Query(pattern="^[a-f0-9]{64}$")] = None
    rating: Annotated[Literal["up", "down", "none"] | None, Query()] = None
    reason: Annotated[FeedbackReason | None, Query()] = None
    has_comment: Annotated[bool | None, Query()] = None
    review_status: Annotated[ReviewStatus | None, Query()] = None
    needs_review: Annotated[bool, Query()] = False
    sort_by: Annotated[Literal["date", "latency", "confidence"], Query()] = "date"
    sort_order: Annotated[Literal["asc", "desc"], Query()] = "desc"
    date_from: Annotated[datetime | None, Query()] = None
    date_to: Annotated[datetime | None, Query()] = None
    search: Annotated[str | None, Query()] = None
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = DEFAULT_PAGE_LIMIT
    offset: Annotated[int, Query(ge=0)] = DEFAULT_PAGE_OFFSET

    def to_filter(self, user_ids: tuple[int, ...] | None) -> ChatLogFilter:
        return ChatLogFilter(**asdict(self), user_ids=user_ids)


EXPORT_FIELDS = (
    *(name for name in ChatLogEntry.model_fields if name not in {"feedback", "review"}),
    "rating",
    "reasons",
    "comment",
    "review_status",
    "review_note",
    "benchmark_question_id",
)


router = APIRouter(tags=["admin-chat-logs"])


@router.get("/admin/chat-logs")
async def list_chat_logs(
    *,
    options: Annotated[ChatLogQuery, Depends()],
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.SYSTEM_OBSERVE))],
    service: Annotated[ChatLogService, Depends(create_chat_log_service)],
    assignment_service: Annotated[AssignmentService, Depends(create_assignment_service)],
) -> ChatLogsResponse:
    filters = options.to_filter(await visible_user_ids(current_user, assignment_service))
    total = await service.count_logs(filters)
    rows = await service.list_rows(filters)
    return ChatLogsResponse(
        logs=[entry_response(row) for row in rows if row.entry.id is not None], total=total
    )


@router.get("/admin/chat-logs/stats")
async def chat_log_stats(
    *,
    options: Annotated[ChatLogQuery, Depends()],
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.SYSTEM_OBSERVE))],
    service: Annotated[ChatLogService, Depends(create_chat_log_service)],
    assignment_service: Annotated[AssignmentService, Depends(create_assignment_service)],
) -> dict[str, Any]:
    """Aggregates over the same filters and curator scope as the list; paging is ignored."""
    return await service.stats(options.to_filter(await visible_user_ids(current_user, assignment_service)))


@router.get("/admin/chat-logs/export", response_model=None)
async def export_chat_logs(
    *,
    options: Annotated[ChatLogQuery, Depends()],
    export_format: Annotated[Literal["csv", "json"], Query(alias="format")] = "csv",
    max_rows: Annotated[int, Query(ge=1, le=5000)] = 5000,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.SYSTEM_OBSERVE))],
    service: Annotated[ChatLogService, Depends(create_chat_log_service)],
    assignment_service: Annotated[AssignmentService, Depends(create_assignment_service)],
) -> Response:
    """Journal and reaction fields only: no traces, fragment text or configuration snapshots."""
    filters = options.to_filter(await visible_user_ids(current_user, assignment_service))
    records = [export_record(row) for row in await service.export_rows(filters, max_rows)]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005 - file name only

    if export_format == "json":
        content = json.dumps(records, ensure_ascii=False, default=str)
        media_type = "application/json"
    else:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        writer.writerows(
            {
                key: csv_cell(
                    json.dumps(value, ensure_ascii=False) if isinstance(value, list | dict) else value
                )
                for key, value in record.items()
            }
            for record in records
        )
        content = buffer.getvalue()
        media_type = "text/csv; charset=utf-8"

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="chat-logs-{stamp}.{export_format}"'},
    )


# Spreadsheets evaluate a cell starting with these characters as a formula.
CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def csv_cell(value: object) -> object:
    """Neutralize formulas in user-controlled text (questions, answers, comments)."""
    if isinstance(value, str) and value.lstrip(" ").startswith(CSV_FORMULA_PREFIXES):
        return f"'{value}"

    return value


def export_record(row: ChatLogRow) -> dict[str, Any]:
    entry = entry_response(row).model_dump(mode="json", exclude={"feedback", "review"})
    return {
        **{key: entry.get(key) for key in EXPORT_FIELDS if key in entry},
        "rating": row.feedback.rating.value if row.feedback else None,
        "reasons": sorted(row.feedback.reasons) if row.feedback else [],
        "comment": row.feedback.comment if row.feedback else None,
        "review_status": row.review.status.value if row.review else ReviewStatus.NEW.value,
        "review_note": row.review.note if row.review else None,
        "benchmark_question_id": row.review.benchmark_question_id if row.review else None,
    }


async def visible_user_ids(
    current_user: CurrentUser, assignment_service: AssignmentService
) -> tuple[int, ...] | None:
    if current_user.role != UserRole.CURATOR:
        return None

    scope = await assignment_service.get_scope(current_user.id)
    return tuple(scope["managed_client_ids"] + scope["managed_internal_ids"] + [current_user.id])


def review_response(review: ChatLogReview) -> ChatLogReviewResponse:
    return ChatLogReviewResponse(**asdict(review))


def entry_response(row: ChatLogRow) -> ChatLogEntry:
    data = asdict(row.entry)
    data["creation_date"] = row.entry.creation_date.isoformat()
    return ChatLogEntry(
        **data,
        user_email=row.email,
        feedback=feedback_response(row.feedback) if row.feedback else None,
        review=review_response(row.review) if row.review else None,
    )


@router.get("/admin/chat-logs/{log_id}")
async def get_chat_log(
    log_id: int,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.SYSTEM_OBSERVE))],
    service: Annotated[ChatLogService, Depends(create_chat_log_service)],
    assignment_service: Annotated[AssignmentService, Depends(create_assignment_service)],
) -> dict[str, Any]:
    details = await service.get_log(log_id, await visible_user_ids(current_user, assignment_service))

    if details is None:
        raise HTTPException(status_code=404, detail="Chat log not found")

    return {
        "entry": entry_response(details.row),
        "snapshot": service.compare_snapshot(details.snapshot),
        "context": [
            {
                "id": turn.id,
                "role": turn.role.value,
                "content": turn.content,
                "creation_date": turn.creation_date,
            }
            for turn in details.context
        ],
    }


@router.put(
    "/admin/chat-logs/{log_id}/review",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def review_chat_log(
    log_id: int,
    body: ChatLogReviewRequest,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.SYSTEM_OBSERVE))],
    service: Annotated[ChatLogService, Depends(create_chat_log_service)],
    assignment_service: Annotated[AssignmentService, Depends(create_assignment_service)],
) -> ChatLogReviewResponse:
    """Triage within the viewer's journal scope; ``in_benchmark`` comes only from a transfer."""
    review = await service.reviews.set_status(
        log_id,
        body.status,
        body.note,
        current_user.id,
        await visible_user_ids(current_user, assignment_service),
    )
    return review_response(review)


@router.get("/admin/chat-logs/{log_id}/benchmark-question")
async def benchmark_question_draft(
    log_id: int,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.BENCHMARK_MANAGE))],
    service: Annotated[ChatLogService, Depends(create_chat_log_service)],
    assignment_service: Annotated[AssignmentService, Depends(create_assignment_service)],
) -> BenchmarkDraftResponse:
    """Redacted question and previous turns for the transfer form to edit."""
    draft = await service.reviews.draft(log_id, await visible_user_ids(current_user, assignment_service))
    return BenchmarkDraftResponse(**asdict(draft))


@router.post(
    "/admin/chat-logs/{log_id}/benchmark-question",
    status_code=201,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def transfer_to_benchmark(
    log_id: int,
    body: BenchmarkTransferRequest,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.BENCHMARK_MANAGE))],
    service: Annotated[ChatLogService, Depends(create_chat_log_service)],
    assignment_service: Annotated[AssignmentService, Depends(create_assignment_service)],
) -> dict[str, Any]:
    question = await service.reviews.transfer(
        log_id,
        BenchmarkTransfer(
            dataset=body.dataset,
            question=body.question,
            history=[turn.model_dump() for turn in body.history],
        ),
        current_user.id,
        await visible_user_ids(current_user, assignment_service),
    )
    return {"question_id": question.id, "dataset": question.dataset}


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
