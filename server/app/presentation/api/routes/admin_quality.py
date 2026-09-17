"""Admin quality / diagnostics endpoints — document extraction quality, dry-run preview."""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

from domain.value_objects.file_backend import FileBackend
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from application.ports.rate_limit import RateLimitPolicyName
from application.services.pdf_diagnostic_service import PDFDiagnosticService
from application.services.quality_service import QualityService
from infrastructure.ml.preview.factory import PreviewStrategyFactory

from presentation.api.auth_dependencies import require_admin
from presentation.api.rate_limit import rate_limit
from presentation.api.constants import FILE_TOO_LARGE_STATUS, PAGE_IMAGE_DPI
from presentation.api.dependencies import (
    create_action_logger,
    create_document_service,
    create_domain_registry,
    create_domain_settings,
    create_job_enqueuer,
    create_job_service,
    create_pdf_diagnostic_service,
    create_preview_cache,
    create_quality_service,
    create_storage_config,
)
from presentation.api.helpers import (
    build_dry_run_response,
    compute_quality_warning,
    upload_and_enqueue,
)
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
router = APIRouter(tags=["admin-quality"])


# ---------------------------------------------------------------------------
# Quality list & diagnosis
# ---------------------------------------------------------------------------


@router.get("/admin/documents/quality", response_model=DocumentQualityListResponse)
async def list_quality_documents(
    admin: CurrentUser = Depends(require_admin),
    quality_service: QualityService = Depends(create_quality_service),
):
    """List documents with quality warnings, sorted by quality_score descending."""
    warned = await quality_service.list_warned_documents()

    return DocumentQualityListResponse(
        documents=[
            DocumentQualityItem(
                id=d.id,
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
    response_model=DocumentDiagnoseResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def diagnose_document(
    document_id: int,
    admin: CurrentUser = Depends(require_admin),
    diag_service: PDFDiagnosticService = Depends(create_pdf_diagnostic_service),
    quality_service: QualityService = Depends(create_quality_service),
):
    """Run per-page PDF diagnosis on an already-indexed document."""
    source_path = await quality_service.get_document_source_path(document_id)

    result = await diag_service.diagnose_document(document_id, source_path)
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to diagnose document")

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
    response_model=DryRunResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def dry_run_preview(
    file: UploadFile = File(...),
    admin: CurrentUser = Depends(require_admin),
    diag_service: PDFDiagnosticService = Depends(create_pdf_diagnostic_service),
    preview_cache=Depends(create_preview_cache),
    domain_registry=Depends(create_domain_registry),
    domain_settings=Depends(create_domain_settings),
    storage_cfg=Depends(create_storage_config),
):
    """Phase 1: Fast dry-run — text layer only, no OCR."""
    filename = file.filename or "unnamed"
    try:
        PreviewFile(filename=filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    data = await file.read()
    if len(data) > diag_service._max_bytes:
        raise HTTPException(
            status_code=FILE_TOO_LARGE_STATUS,
            detail="File too large for dry-run (max 50 MB)",
        )

    ext = Path(filename).suffix.lower()

    try:
        strategy = PreviewStrategyFactory.for_extension(
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
            image_available=storage_cfg.file_backend == FileBackend.S3.value,
        )


# ---------------------------------------------------------------------------
# Dry-run preview — Phase 2 (OCR)
# ---------------------------------------------------------------------------


@router.post(
    "/admin/documents/preview-ocr",
    response_model=DryRunResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def dry_run_ocr_phase2(
    file: UploadFile = File(None),
    preview_id: str = Form(""),
    pages: str = Form(""),
    admin: CurrentUser = Depends(require_admin),
    diag_service: PDFDiagnosticService = Depends(create_pdf_diagnostic_service),
    preview_cache=Depends(create_preview_cache),
    domain_registry=Depends(create_domain_registry),
    domain_settings=Depends(create_domain_settings),
    storage_cfg=Depends(create_storage_config),
):
    """Phase 2: Run OCR on specific problem units and return updated results.

    Accepts either a fresh file upload OR a ``preview_id`` from a previous
    ``/preview`` call (avoids re-uploading the file from the client).
    """
    try:
        page_nums = [int(p.strip()) for p in pages.split(",") if p.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid unit IDs") from None

    if not page_nums:
        raise HTTPException(status_code=400, detail="No units specified for OCR")

    async def _run_ocr(tmp_path: Path, effective_preview_id: str, fname: str) -> DryRunResponse:
        ext = Path(fname).suffix.lower()
        try:
            strategy = PreviewStrategyFactory.for_extension(
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
            image_available=storage_cfg.file_backend == FileBackend.S3.value,
        )

    # Resolve the file: cached file takes precedence over a fresh upload
    if preview_id:
        async with preview_cache.get_path(preview_id) as cached_path:
            if cached_path is not None:
                fname = preview_cache.get_filename(preview_id)
                return await _run_ocr(cached_path, preview_id, fname)

    # No cached file — require a fresh upload
    if file is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file upload or a valid preview_id",
        )
    return await _run_ocr_from_upload(file, diag_service, preview_cache, _run_ocr)


async def _run_ocr_from_upload(
    file: UploadFile,
    diag_service: PDFDiagnosticService,
    preview_cache,
    run_ocr_fn,
) -> DryRunResponse:
    """Handle fresh file upload for OCR: validate, cache, run OCR."""
    fname = file.filename or "unnamed"
    try:
        PreviewFile(filename=fname)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    data = await file.read()
    if len(data) > diag_service._max_bytes:
        raise HTTPException(
            status_code=FILE_TOO_LARGE_STATUS,
            detail="File too large for dry-run (max 50 MB)",
        )

    ext = Path(fname).suffix.lower()
    new_preview_id = await preview_cache.store(data, suffix=ext, original_filename=fname)

    async with preview_cache.get_path(new_preview_id) as tmp_path:
        if tmp_path is None:
            raise HTTPException(status_code=500, detail="Failed to store preview")
        return await run_ocr_fn(tmp_path, new_preview_id, fname)


# ---------------------------------------------------------------------------
# Page image rendering (PDF only)
# ---------------------------------------------------------------------------


def _render_page_image_sync(diag_service: PDFDiagnosticService, tmp_path: Path, page: int) -> bytes:
    """Sync PyMuPDF work (open/count/render/close) — executed in a worker thread."""
    doc = diag_service._pdf.open(str(tmp_path))
    try:
        total = diag_service._pdf.get_page_count(doc)
        if page > total:
            raise HTTPException(
                status_code=400,
                detail=f"Page {page} out of range (document has {total} pages)",
            )
        return diag_service._pdf.render_page_image(doc, page - 1, dpi=PAGE_IMAGE_DPI)
    finally:
        diag_service._pdf.close(doc)


@router.post(
    "/admin/documents/preview/page-image",
    response_model=PageImageResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def get_page_image(
    preview_id: str = Form(...),
    page: int = Form(...),
    admin: CurrentUser = Depends(require_admin),
    diag_service: PDFDiagnosticService = Depends(create_pdf_diagnostic_service),
    preview_cache=Depends(create_preview_cache),
    storage_cfg=Depends(create_storage_config),
):
    """Render a single page of the cached PDF as a PNG image (base64-encoded)."""
    if storage_cfg.file_backend != FileBackend.S3.value:
        raise HTTPException(
            status_code=404,
            detail="Page image rendering is not available (requires S3 storage backend)",
        )

    async with preview_cache.get_path(preview_id) as tmp_path:
        if tmp_path is None:
            raise HTTPException(status_code=404, detail="Preview expired or not found")

        if page < 1:
            raise HTTPException(status_code=400, detail="Page number must be >= 1")

        image_bytes = await asyncio.to_thread(_render_page_image_sync, diag_service, tmp_path, page)

        return PageImageResponse(
            image_base64=base64.b64encode(image_bytes).decode("ascii"),
            page=page,
        )


# ---------------------------------------------------------------------------
# Index directly from dry-run
# ---------------------------------------------------------------------------


@router.post(
    "/admin/documents/preview/{preview_id}/index",
    response_model=IndexFromPreviewResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def index_from_preview(
    preview_id: str,
    visibility: str = Form("internal_public"),
    group_id: int | None = Form(None),
    client_id: int | None = Form(None),
    doc_domain: str | None = Form(None),
    admin: CurrentUser = Depends(require_admin),
    preview_cache=Depends(create_preview_cache),
    document_service=Depends(create_document_service),
    job_service=Depends(create_job_service),
    job_enqueuer=Depends(create_job_enqueuer),
    log=Depends(create_action_logger),
):
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
        visibility=visibility,
        group_id=group_id,
        client_id=client_id,
        user_id=admin.id,
        user_kind=admin.kind,
        user_role=admin.role,
        rename_on_conflict=False,
        doc_domain=doc_domain,
        document_service=document_service,
        job_service=job_service,
        enqueue_fn=job_enqueuer.enqueue_document_processing,
        action_name="document.upload_from_preview",
        log_fn=log,
    )

    return {"document_id": result["document_id"], "filename": filename, "status": result["status"]}
