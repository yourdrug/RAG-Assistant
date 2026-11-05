"""Admin quality / diagnostics endpoints — document extraction quality, dry-run preview."""

import asyncio
import base64
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Annotated, cast

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from application.documents import DocumentCommandService, PreviewCache, PreviewStrategyFactoryPort
from application.documents.diagnostics import PDFDiagnosticService
from application.documents.quality_service import QualityService
from application.ports.rate_limit import RateLimitPolicyName
from domain.ingestion.ports import DomainSettingsPort
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.knowledge.enums import DocDomain, DocumentVisibility
from domain.shared.exceptions import AppServerError
from presentation.api.auth_dependencies import require_admin
from presentation.api.constants import PAGE_IMAGE_DPI
from presentation.api.dependencies import (
    JobSubmission,
    create_document_command_service,
    create_domain_registry,
    create_domain_settings,
    create_job_submission,
    create_pdf_diagnostic_service,
    create_preview_cache,
    create_preview_strategy_factory,
    create_quality_service,
)
from presentation.api.helpers import (
    build_dry_run_response,
    compute_quality_warning,
    persisted_id,
    read_upload_with_limit,
    upload_and_enqueue,
)
from presentation.api.rate_limit import rate_limit
from presentation.api.schemas import (
    CurrentUser,
    DocumentDiagnoseResponse,
    DocumentQualityItem,
    DocumentQualityListResponse,
    DryRunResponse,
    IndexFromPreviewResponse,
    PageDiagnostic,
    PageImageResponse,
    PreviewFile,
)


logger = logging.getLogger("default")


# Preserve the legacy optional file schema without adding a nullable wire type.
DEFAULT_PREVIEW_FILE = cast("UploadFile", None)


@dataclass(kw_only=True)
class PreviewOCRForm:
    file: Annotated[UploadFile, File()] = DEFAULT_PREVIEW_FILE
    preview_id: Annotated[str, Form()] = ""
    pages: Annotated[str, Form()] = ""


@dataclass(kw_only=True)
class PreviewIndexForm:
    visibility: Annotated[str, Form()] = DocumentVisibility.INTERNAL_PUBLIC.value
    group_id: Annotated[int | None, Form()] = None
    client_id: Annotated[int | None, Form()] = None
    doc_domain: Annotated[DocDomain | None, Form()] = None


router = APIRouter(tags=["admin-quality"])


# ---------------------------------------------------------------------------
# Quality list & diagnosis
# ---------------------------------------------------------------------------


@router.get("/admin/documents/quality")
async def list_quality_documents(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    quality_service: Annotated[QualityService, Depends(create_quality_service)],
) -> DocumentQualityListResponse:
    """List documents with quality warnings, sorted by quality_score descending."""
    warned = await quality_service.list_warned_documents()

    return DocumentQualityListResponse(
        documents=[
            DocumentQualityItem(
                id=persisted_id(d.id),
                filename=d.filename,
                status=d.status,
                quality_score=d.quality_score,
                warning_message=d.warning_message,
                chunks=d.chunks,
                chars=d.chars,
                indexed_at=d.indexed_at,
            )
            for d in warned
        ],
        total=len(warned),
    )


@router.post(
    "/admin/documents/{document_id}/diagnose",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def diagnose_document(
    *,
    document_id: int,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    diag_service: Annotated[PDFDiagnosticService, Depends(create_pdf_diagnostic_service)],
    quality_service: Annotated[QualityService, Depends(create_quality_service)],
) -> DocumentDiagnoseResponse:
    """Run per-page PDF diagnosis on an already-indexed document."""
    source_path = await quality_service.get_document_source_path(document_id)

    result = await diag_service.diagnose_document(document_id, source_path)

    if result is None:
        raise AppServerError(message="Failed to diagnose document")

    return DocumentDiagnoseResponse(
        document_id=result.document_id,
        filename=result.filename,
        total_pages=result.total_pages,
        pages=[
            PageDiagnostic(page=p.page, type=p.type, chars=p.chars, description=p.description)
            for p in result.pages
        ],
        summary=result.summary,
    )


# ---------------------------------------------------------------------------
# Dry-run preview — Phase 1
# ---------------------------------------------------------------------------


@router.post(
    "/admin/documents/preview",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def dry_run_preview(
    *,
    file: Annotated[UploadFile, File()],
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    diag_service: Annotated[PDFDiagnosticService, Depends(create_pdf_diagnostic_service)],
    preview_cache: Annotated[PreviewCache, Depends(create_preview_cache)],
    preview_factory: Annotated[PreviewStrategyFactoryPort, Depends(create_preview_strategy_factory)],
    domain_registry: Annotated[DomainProfileRegistry | None, Depends(create_domain_registry)],
    domain_settings: Annotated[DomainSettingsPort | None, Depends(create_domain_settings)],
) -> DryRunResponse:
    """Preview extraction; the selected PaddleOCR-VL engine also parses scans."""
    filename = file.filename or "unnamed"
    try:
        PreviewFile(filename=filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    data = await read_upload_with_limit(file, diag_service.max_preview_bytes)

    ext = Path(filename).suffix.lower()

    try:
        strategy = preview_factory.for_extension(
            ext,
            diag_service=diag_service,
            domain_registry=domain_registry,
            domain_settings=domain_settings,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    preview_id = await preview_cache.store(data, suffix=ext, original_filename=filename)

    async with preview_cache.get_path(preview_id) as tmp_path:
        if tmp_path is None:
            raise HTTPException(status_code=500, detail="Failed to store preview")

        # PyMuPDF over every page is CPU-heavy: run in a worker thread so the
        # event loop (single uvicorn process) stays responsive.
        page_results, types_count, total_chars = await asyncio.to_thread(strategy.analyze, tmp_path)
        warning = compute_quality_warning(page_results, types_count, "Low quality")
        suggestion = PDFDiagnosticService.suggest_action(page_results, types_count)
        return build_dry_run_response(
            filename,
            page_results,
            types_count,
            total_chars,
            warning,
            preview_id=preview_id,
            suggestion=suggestion,
            image_available=True,
        )


# ---------------------------------------------------------------------------
# Dry-run preview — Phase 2 (OCR)
# ---------------------------------------------------------------------------


@router.post(
    "/admin/documents/preview-ocr",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def dry_run_ocr_phase2(
    *,
    options: Annotated[PreviewOCRForm, Depends()],
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    diag_service: Annotated[PDFDiagnosticService, Depends(create_pdf_diagnostic_service)],
    preview_cache: Annotated[PreviewCache, Depends(create_preview_cache)],
    preview_factory: Annotated[PreviewStrategyFactoryPort, Depends(create_preview_strategy_factory)],
    domain_registry: Annotated[DomainProfileRegistry | None, Depends(create_domain_registry)],
    domain_settings: Annotated[DomainSettingsPort | None, Depends(create_domain_settings)],
) -> DryRunResponse:
    """Phase 2: Run OCR on specific problem units and return updated results.

    Accepts either a fresh file upload OR a ``preview_id`` from a previous
    ``/preview`` call (avoids re-uploading the file from the client).
    """
    page_nums = parse_ocr_pages(options.pages)

    async def _run_ocr(tmp_path: Path, effective_preview_id: str, fname: str) -> DryRunResponse:
        ext = Path(fname).suffix.lower()
        try:
            strategy = preview_factory.for_extension(
                ext,
                diag_service=diag_service,
                domain_registry=domain_registry,
                domain_settings=domain_settings,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        page_results, types_count, total_chars = await asyncio.to_thread(strategy.analyze, tmp_path)
        try:
            page_results, types_count, total_chars = await asyncio.to_thread(
                strategy.ocr_problem_units, tmp_path, page_results, page_nums
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        warning = compute_quality_warning(page_results, types_count, "Still low quality after OCR")
        suggestion = PDFDiagnosticService.suggest_action(page_results, types_count)
        return build_dry_run_response(
            fname,
            page_results,
            types_count,
            total_chars,
            warning,
            preview_id=effective_preview_id,
            suggestion=suggestion,
            image_available=True,
        )

    # Resolve the file: cached file takes precedence over a fresh upload
    if options.preview_id:
        async with preview_cache.get_path(options.preview_id) as cached_path:
            if cached_path is not None:
                fname = preview_cache.get_filename(options.preview_id)
                return await _run_ocr(cached_path, options.preview_id, fname)

    # No cached file — require a fresh upload
    if options.file is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file upload or a valid preview_id",
        )

    return await run_ocr_from_upload(options.file, diag_service, preview_cache, _run_ocr)


async def run_ocr_from_upload(
    file: UploadFile,
    diag_service: PDFDiagnosticService,
    preview_cache: PreviewCache,
    run_ocr_fn: Callable[[Path, str, str], Awaitable[DryRunResponse]],
) -> DryRunResponse:
    """Handle fresh file upload for OCR: validate, cache, run OCR."""
    fname = file.filename or "unnamed"
    try:
        PreviewFile(filename=fname)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    data = await read_upload_with_limit(file, diag_service.max_preview_bytes)

    ext = Path(fname).suffix.lower()
    new_preview_id = await preview_cache.store(data, suffix=ext, original_filename=fname)

    async with preview_cache.get_path(new_preview_id) as tmp_path:
        if tmp_path is None:
            raise HTTPException(status_code=500, detail="Failed to store preview")

        return await run_ocr_fn(tmp_path, new_preview_id, fname)


# ---------------------------------------------------------------------------
# Page image rendering (PDF only)
# ---------------------------------------------------------------------------


def render_page_image_sync(diag_service: PDFDiagnosticService, tmp_path: Path, page: int) -> bytes:
    """Sync PyMuPDF work (open/count/render/close) — executed in a worker thread."""
    doc = diag_service.pdf_document.open(str(tmp_path))
    try:
        total = diag_service.pdf_document.get_page_count(doc)

        if page > total:
            raise HTTPException(
                status_code=400,
                detail=f"Page {page} out of range (document has {total} pages)",
            )

        return diag_service.pdf_document.render_page_image(doc, page - 1, dpi=PAGE_IMAGE_DPI)
    finally:
        diag_service.pdf_document.close(doc)


@router.post(
    "/admin/documents/preview/page-image",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def get_page_image(
    *,
    preview_id: Annotated[str, Form()],
    page: Annotated[int, Form()],
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    diag_service: Annotated[PDFDiagnosticService, Depends(create_pdf_diagnostic_service)],
    preview_cache: Annotated[PreviewCache, Depends(create_preview_cache)],
) -> PageImageResponse:
    """Render a single page of the cached PDF as a PNG image (base64-encoded)."""
    async with preview_cache.get_path(preview_id) as tmp_path:
        if tmp_path is None:
            raise HTTPException(status_code=404, detail="Preview expired or not found")

        if page < 1:
            raise HTTPException(status_code=400, detail="Page number must be >= 1")

        image_bytes = await asyncio.to_thread(render_page_image_sync, diag_service, tmp_path, page)

        return PageImageResponse(
            image_base64=base64.b64encode(image_bytes).decode("ascii"),
            page=page,
        )


# ---------------------------------------------------------------------------
# Index directly from dry-run
# ---------------------------------------------------------------------------


@router.post(
    "/admin/documents/preview/{preview_id}/index",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def index_from_preview(
    *,
    preview_id: str,
    options: Annotated[PreviewIndexForm, Depends()],
    admin: Annotated[CurrentUser, Depends(require_admin)],
    preview_cache: Annotated[PreviewCache, Depends(create_preview_cache)],
    cmd: Annotated[DocumentCommandService, Depends(create_document_command_service)],
    submission: Annotated[JobSubmission, Depends(create_job_submission)],
) -> IndexFromPreviewResponse:
    """Index the cached document through the standard ingestion pipeline.

    The uploaded file is passed through the same upload + processing flow as
    a normal document upload.  Any OCR results from the dry-run are *not*
    reused — the standard ``DocumentProcessor`` re-processes the file from
    scratch.
    """
    file_data = await preview_cache.get_bytes(preview_id)

    if file_data is None:
        raise HTTPException(status_code=404, detail="Preview expired or not found")

    filename = preview_cache.get_filename(preview_id)

    result = await upload_and_enqueue(
        file_data=file_data,
        filename=filename,
        visibility=options.visibility,
        group_id=options.group_id,
        client_id=options.client_id,
        user_id=admin.id,
        user_kind=admin.kind,
        user_role=admin.role,
        rename_on_conflict=False,
        doc_domain=options.doc_domain.value if options.doc_domain is not None else None,
        document_service=cmd,
        action_name="document.upload_from_preview",
        log_fn=submission.log,
    )

    return IndexFromPreviewResponse(
        document_id=result["document_id"], filename=filename, status=result["status"]
    )


def parse_ocr_pages(pages: str) -> list[int]:
    try:
        page_nums = [int(p.strip()) for p in pages.split(",") if p.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid unit IDs") from None

    if not page_nums:
        raise HTTPException(status_code=400, detail="No units specified for OCR")

    return page_nums
