"""Reusable presentation helpers — validation, file upload, response building.

Consolidates duplicated validation, upload-read, quality-warning, and
response-building patterns that were previously scattered across route handlers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, NotRequired, Protocol, TypedDict, Unpack, runtime_checkable

from fastapi import HTTPException

from application.benchmark import BenchmarkQuestionCreateDTO, SweepCreateDTO
from application.benchmark.dto import BenchmarkResultSummary as ResultSummary
from application.documents.dto import UploadOptions
from domain.benchmark.question import BenchmarkQuestion
from domain.benchmark.run import BenchmarkRun
from domain.benchmark.sweep import BenchmarkSweep
from domain.ingestion.preview import DryRunPageResult as PreviewPage
from domain.knowledge.enums import DocumentStatus
from presentation.api.constants import (
    PAGE_PREVIEW_MAX_CHARS,
    QUALITY_BAD_RATIO_THRESHOLD,
)
from presentation.api.schemas import (
    BenchmarkQuestionCreate,
    BenchmarkQuestionResponse,
    BenchmarkResultSummary,
    BenchmarkRunResponse,
    DryRunPageResult,
    DryRunResponse,
    SweepCreateRequest,
    SweepResponse,
)


if TYPE_CHECKING:
    from fastapi import UploadFile

    from application.documents import DocumentDTO
    from application.ports.action_logger import ActionLoggerPort

logger = logging.getLogger("default")

CHUNK_READ_SIZE = 64 * 1024  # 64 KB per read iteration


class UploadResult(TypedDict):
    """Typed result of upload_and_enqueue — keeps route responses type-safe."""

    document_id: int
    filename: str
    status: str


# ---------------------------------------------------------------------------
# Source filtering helper (strip internal metadata keys)
# ---------------------------------------------------------------------------


def filter_sources(
    sources: list[dict[str, Any]] | None, *, exclude_keys: frozenset[str]
) -> list[dict[str, Any]] | None:
    """Remove internal metadata keys (e.g. ``_confidence``) from source dicts."""
    if not sources:
        return None

    return [s for s in sources if not any(k in s for k in exclude_keys)]


# ---------------------------------------------------------------------------
# Document upload + job enqueue orchestration
# ---------------------------------------------------------------------------


@runtime_checkable
class UploadableService(Protocol):
    async def upload(
        self,
        filename: str,
        file_data: bytes,
        visibility: str,
        group_id: int | None,
        **options: Unpack[UploadOptions],
    ) -> DocumentDTO: ...


class UploadRequest(TypedDict):
    file_data: bytes
    filename: str
    visibility: str
    group_id: int | None
    client_id: int | None
    user_id: int
    user_kind: str
    user_role: str
    rename_on_conflict: bool
    doc_domain: str | None
    replaces_document_id: NotRequired[int | None]


async def upload_and_enqueue(
    *,
    document_service: UploadableService,
    action_name: str,
    log_fn: ActionLoggerPort | None = None,
    **upload: Unpack[UploadRequest],
) -> UploadResult:
    """Upload a document and its durable processing job in one transaction.

    Returns a dict with ``document_id``, ``filename``, and ``status``.
    """
    result = await document_service.upload(
        filename=upload["filename"],
        file_data=upload["file_data"],
        visibility=upload["visibility"],
        group_id=upload["group_id"],
        client_id=upload["client_id"],
        user_id=upload["user_id"],
        user_kind=upload["user_kind"],
        user_role=upload["user_role"],
        rename_on_conflict=upload["rename_on_conflict"],
        doc_domain=upload["doc_domain"],
        replaces_document_id=upload.get("replaces_document_id"),
    )

    if log_fn is not None:
        log_fn(
            action_name,
            user_id=upload["user_id"],
            details={"filename": upload["filename"], "visibility": upload["visibility"]},
        )

    return {
        "document_id": result.id,
        "filename": upload["filename"],
        "status": DocumentStatus.PROCESSING.value,
    }


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------


async def read_upload_with_limit(file: UploadFile, max_bytes: int) -> bytes:
    """Read file content in chunks up to max_bytes; raise 413 if exceeded."""
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await file.read(CHUNK_READ_SIZE)

        if not chunk:
            break

        total += len(chunk)

        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: >{max_bytes / 1024 / 1024:.0f} MB limit",
            )

        chunks.append(chunk)

    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Path traversal prevention
# ---------------------------------------------------------------------------


def validate_data_path_within_dir(path_str: str, data_dir: str) -> str:
    """Resolve path and ensure it's within data_dir to prevent path traversal.

    Raises:
        HTTPException: 400 if path escapes data_dir.

    """
    resolved = Path(path_str).resolve()
    data_root = Path(data_dir).resolve()

    if not str(resolved).startswith(str(data_root) + "/") and resolved != data_root:
        raise HTTPException(status_code=400, detail="Path must be within the data directory")

    return str(resolved)


# ---------------------------------------------------------------------------
# Dry-run quality analysis
# ---------------------------------------------------------------------------


def compute_quality_warning(
    page_results: list[PreviewPage], types_count: dict[str, int], warning_prefix: str
) -> str | None:
    """Compute quality warning from page analysis results."""
    total = len(page_results)
    bad = types_count.get("scan", 0) + types_count.get("garbled", 0) + types_count.get("empty", 0)
    bad_ratio = bad / total if total else 0.0

    if bad_ratio > QUALITY_BAD_RATIO_THRESHOLD:
        return (
            f"{warning_prefix}: {types_count.get('scan', 0)} scan + "
            f"{types_count.get('garbled', 0)} garbled + "
            f"{types_count.get('empty', 0)} empty "
            f"out of {total} units ({bad_ratio:.0%} bad)"
        )

    return None


class DryRunOptions(TypedDict, total=False):
    preview_id: str | None
    suggestion: str | None
    image_available: bool


def build_dry_run_response(
    filename: str,
    page_results: list[PreviewPage],
    types_count: dict[str, int],
    total_chars: int,
    warning: str | None,
    **options: Unpack[DryRunOptions],
) -> DryRunResponse:
    """Build the public preview response from page analysis results."""
    total_pages = len(page_results)
    bad = types_count.get("scan", 0) + types_count.get("garbled", 0) + types_count.get("empty", 0)
    bad_ratio = bad / total_pages if total_pages else 0.0
    text_previews = (p.preview for p in page_results if p.type == "text")
    full_text_preview = "\n\n".join(text_previews)[:PAGE_PREVIEW_MAX_CHARS]
    return DryRunResponse(
        filename=filename,
        total_pages=total_pages,
        pages=[
            DryRunPageResult(
                page=p.page,
                type=p.type,
                content_type=p.content_type,
                chars=p.chars,
                preview=p.preview,
                full_text=p.full_text,
                problem_spans=p.problem_spans,
                previous_type=p.previous_type,
                image_available=options.get("image_available", False) and p.unit_kind == "page",
                unit_kind=p.unit_kind,
                label=p.label,
            )
            for p in page_results
        ],
        total_chars=total_chars,
        quality_score=bad_ratio,
        warning=warning,
        full_text_preview=full_text_preview,
        summary=types_count,
        preview_id=options.get("preview_id"),
        suggestion=options.get("suggestion"),
    )


# ---------------------------------------------------------------------------
# Benchmark DTO ↔ Schema mappers
# ---------------------------------------------------------------------------


def question_create_to_dto(body: BenchmarkQuestionCreate) -> BenchmarkQuestionCreateDTO:
    """Map ``BenchmarkQuestionCreate`` schema → ``BenchmarkQuestionCreateDTO``."""
    return BenchmarkQuestionCreateDTO(
        question=body.question,
        expected_answer=body.expected_answer,
        source_hint=body.source_hint,
        annotations=body.annotations,
        case_id=body.case_id,
        variant_of=body.variant_of,
        question_style=body.question_style,
        history=body.history,
        as_of_date=body.as_of_date,
        tags=body.tags,
        dataset=body.dataset,
        notes=body.notes,
        is_active=body.is_active,
    )


def sweep_create_to_dto(body: SweepCreateRequest) -> SweepCreateDTO:
    """Map ``SweepCreateRequest`` schema → ``SweepCreateDTO``."""
    return SweepCreateDTO(
        strategy=body.strategy,
        search_space=body.search_space,
        objective_weights=body.objective_weights,
        dataset=body.dataset,
        top_n_llm=body.top_n_llm,
        judge_model=body.judge_model,
        evaluation_mode=body.evaluation_mode,
    )


def question_to_response(q: BenchmarkQuestion) -> BenchmarkQuestionResponse:
    """Map ``BenchmarkQuestion`` entity → ``BenchmarkQuestionResponse`` schema."""
    if q.id is None:
        message = "BenchmarkQuestion saved with None id"
        raise RuntimeError(message)

    return BenchmarkQuestionResponse(
        id=q.id,
        question=q.question,
        expected_answer=q.expected_answer,
        source_hint=q.source_hint,
        annotations=q.annotations,
        case_id=q.case_id,
        variant_of=q.variant_of,
        question_style=q.question_style,
        history=q.history,
        as_of_date=q.as_of_date,
        tags=q.tags,
        dataset=q.dataset,
        is_active=q.is_active,
        created_by=q.created_by,
        notes=q.notes,
        creation_date=q.creation_date,
    )


def run_to_response(r: BenchmarkRun) -> BenchmarkRunResponse:
    """Map ``BenchmarkRun`` entity → ``BenchmarkRunResponse`` schema."""
    if r.id is None:
        message = "BenchmarkRun saved with None id"
        raise RuntimeError(message)

    return BenchmarkRunResponse(
        id=r.id,
        sweep_id=r.sweep_id,
        config_json=r.config_json,
        summary_metrics=r.summary_metrics,
        duration_sec=r.duration_sec,
        llm_evaluated=r.llm_evaluated,
        dataset=r.dataset,
        filename=r.filename,
        creation_date=r.creation_date,
    )


def sweep_to_response(s: BenchmarkSweep, *, job_id: int | None = None) -> SweepResponse:
    """Map ``BenchmarkSweep`` entity → ``SweepResponse`` schema."""
    if s.id is None:
        message = "BenchmarkSweep saved with None id"
        raise RuntimeError(message)

    return SweepResponse(
        id=s.id,
        status=s.status,
        strategy=s.strategy,
        search_space=s.search_space,
        objective_weights=s.objective_weights,
        dataset=s.dataset,
        top_n_llm=s.top_n_llm,
        judge_model=s.judge_model,
        evaluation_mode=s.evaluation_mode,
        total_configs=s.total_configs,
        evaluated_configs=s.evaluated_configs,
        best_run_id=s.best_run_id,
        job_id=job_id if job_id is not None else s.job_id,
        creation_date=s.creation_date,
    )


def summary_to_response(s: ResultSummary) -> BenchmarkResultSummary:
    """Map ``BenchmarkResultSummary`` DTO → ``BenchmarkResultSummary`` schema."""
    return BenchmarkResultSummary(
        id=s.id,
        config_json=s.config_json,
        summary_metrics=s.summary_metrics,
        duration_sec=s.duration_sec,
        llm_evaluated=s.llm_evaluated,
        dataset=s.dataset,
        sweep_id=s.sweep_id,
        creation_date=s.creation_date,
        artifacts=s.artifacts,
    )


def persisted_id(value: int | None) -> int:
    if value is None:
        message = "Cannot serialize an entity without a persisted ID"
        raise RuntimeError(message)

    return value
