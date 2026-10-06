"""Reusable presentation helpers — validation, file upload, response building.

Consolidates duplicated validation, upload-read, quality-warning, and
response-building patterns that were previously scattered across route handlers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

from domain.value_objects.document_status import DocumentStatus
from fastapi import HTTPException

from presentation.api.constants import (
    JobType,
    PAGE_PREVIEW_MAX_CHARS,
    QUALITY_BAD_RATIO_THRESHOLD,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from application.dto.document_dto import DocumentDTO
    from application.ports.action_logger import ActionLoggerPort
    from fastapi import UploadFile

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


def filter_sources(sources: list | None, *, exclude_keys: frozenset[str]) -> list | None:
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
        user_id: int,
        user_kind: str,
        user_role: str,
        client_id: int | None = ...,
        rename_on_conflict: bool = ...,
        doc_domain: str | None = ...,
        replaces_document_id: int | None = ...,
    ) -> "DocumentDTO": ...


@runtime_checkable
class JobServiceProtocol(Protocol):
    async def create_job(self, job_type: str, *, related_id: int | None = None) -> int: ...


async def upload_and_enqueue(
    *,
    file_data: bytes,
    filename: str,
    visibility: str,
    group_id: int | None,
    client_id: int | None,
    user_id: int,
    user_kind: str,
    user_role: str,
    rename_on_conflict: bool,
    doc_domain: str | None,
    replaces_document_id: int | None = None,
    document_service: UploadableService,
    job_service: JobServiceProtocol,
    enqueue_fn: "Callable[..., Awaitable[None]]",
    action_name: str,
    log_fn: "ActionLoggerPort | None" = None,
) -> UploadResult:
    """Shared upload → job-create → enqueue logic used by multiple routes.

    Returns a dict with ``document_id``, ``filename``, and ``status``.
    """
    result = await document_service.upload(
        filename=filename,
        file_data=file_data,
        visibility=visibility,
        group_id=group_id,
        client_id=client_id,
        user_id=user_id,
        user_kind=user_kind,
        user_role=user_role,
        rename_on_conflict=rename_on_conflict,
        doc_domain=doc_domain,
        replaces_document_id=replaces_document_id,
    )

    if log_fn is not None:
        log_fn(action_name, user_id=user_id, details={"filename": filename, "visibility": visibility})

    job_id = await job_service.create_job(JobType.DOCUMENT_PROCESSING, related_id=result.id)

    await enqueue_fn(
        document_id=result.id,
        storage_key=result.storage_key or "",
        filename=result.filename,
        visibility=visibility,
        owner_id=result.owner_id,
        group_id=group_id,
        replace_id=result.replace_id,
        doc_domain=doc_domain,
        job_id=job_id,
        principal_id=user_id,
    )

    return {
        "document_id": result.id,
        "filename": filename,
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


def compute_quality_warning(page_results: list, types_count: dict, warning_prefix: str) -> str | None:
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


def build_dry_run_response(
    filename: str,
    page_results: list,
    types_count: dict,
    total_chars: int,
    warning: str | None,
    *,
    preview_id: str | None = None,
    suggestion: str | None = None,
    image_available: bool = False,
) -> Any:
    """Build DryRunResponse from page analysis results.

    Imports ``DryRunResponse`` and ``DryRunPageResult`` from schemas to avoid
    circular imports at module level.
    """
    from presentation.api.schemas import DryRunPageResult, DryRunResponse

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
                image_available=image_available and p.unit_kind == "page",
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
        preview_id=preview_id,
        suggestion=suggestion,
    )


# ---------------------------------------------------------------------------
# Benchmark DTO ↔ Schema mappers
# ---------------------------------------------------------------------------


def question_create_to_dto(body: Any) -> Any:
    """Map ``BenchmarkQuestionCreate`` schema → ``BenchmarkQuestionCreateDTO``."""
    from application.dto.benchmark_dto import BenchmarkQuestionCreateDTO

    return BenchmarkQuestionCreateDTO(
        question=body.question,
        expected_answer=body.expected_answer,
        source_hint=body.source_hint,
        annotations=body.annotations,
        tags=body.tags,
        dataset=body.dataset,
        notes=body.notes,
        is_active=body.is_active,
    )


def sweep_create_to_dto(body: Any) -> Any:
    """Map ``SweepCreateRequest`` schema → ``SweepCreateDTO``."""
    from application.dto.benchmark_dto import SweepCreateDTO

    return SweepCreateDTO(
        strategy=body.strategy,
        search_space=body.search_space,
        objective_weights=body.objective_weights,
        dataset=body.dataset,
        top_n_llm=body.top_n_llm,
        judge_model=body.judge_model,
        evaluation_mode=body.evaluation_mode,
    )


def question_to_response(q: Any) -> Any:
    """Map ``BenchmarkQuestion`` entity → ``BenchmarkQuestionResponse`` schema."""
    from presentation.api.schemas import BenchmarkQuestionResponse

    if q.id is None:
        raise RuntimeError("BenchmarkQuestion saved with None id")
    return BenchmarkQuestionResponse(
        id=q.id,
        question=q.question,
        expected_answer=q.expected_answer,
        source_hint=q.source_hint,
        annotations=q.annotations,
        tags=q.tags,
        dataset=q.dataset,
        is_active=q.is_active,
        created_by=q.created_by,
        notes=q.notes,
        creation_date=q.creation_date,
    )


def run_to_response(r: Any) -> Any:
    """Map ``BenchmarkRun`` entity → ``BenchmarkRunResponse`` schema."""
    from presentation.api.schemas import BenchmarkRunResponse

    if r.id is None:
        raise RuntimeError("BenchmarkRun saved with None id")
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


def sweep_to_response(s: Any, *, job_id: int | None = None) -> Any:
    """Map ``BenchmarkSweep`` entity → ``SweepResponse`` schema."""
    from presentation.api.schemas import SweepResponse

    if s.id is None:
        raise RuntimeError("BenchmarkSweep saved with None id")
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


def summary_to_response(s: Any) -> Any:
    """Map ``BenchmarkResultSummary`` DTO → ``BenchmarkResultSummary`` schema."""
    from presentation.api.schemas import BenchmarkResultSummary

    return BenchmarkResultSummary(
        id=s.id,
        config_json=s.config_json,
        summary_metrics=s.summary_metrics,
        duration_sec=s.duration_sec,
        llm_evaluated=s.llm_evaluated,
        dataset=s.dataset,
        sweep_id=s.sweep_id,
        creation_date=s.creation_date,
    )
