"""Admin quality / diagnostics endpoints — document extraction quality, dry-run preview."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from config import settings
from domain.value_objects.file_backend import FileBackend
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from application.services.pdf_diagnostic_service import PDFDiagnosticService
from application.services.quality_service import QualityService
from infrastructure.ml.preview.factory import PreviewStrategyFactory
from infrastructure.worker.queue import enqueue_document_processing

from presentation.api.auth_dependencies import require_admin
from presentation.api.constants import (
    FILE_TOO_LARGE_STATUS,
    PAGE_IMAGE_DPI,
    PAGE_PREVIEW_MAX_CHARS,
    QUALITY_BAD_RATIO_THRESHOLD,
)
from presentation.api.dependencies import (
    create_document_service,
    create_job_service,
    create_pdf_diagnostic_service,
    create_preview_cache,
    create_quality_service,
)
from presentation.api.helpers import upload_and_enqueue
from presentation.api.schemas import (
    DocumentDiagnoseResponse,
    DocumentQualityItem,
    DocumentQualityListResponse,
    DryRunPageResult,
    DryRunResponse,
    PageDiagnostic,
    PageImageResponse,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["admin-quality"])

IMAGE_AVAILABLE = settings.file_backend == FileBackend.S3.value

_DRY_RUN_EXTENSIONS = {".pdf", ".docx", ".doc", ".rtf"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _validate_upload(file: UploadFile, diag_service: PDFDiagnosticService) -> bytes:
    """Validate uploaded file is a supported format and within size limits. Returns file data."""
    filename = file.filename or "unnamed"
    ext = Path(filename).suffix.lower()
    if ext not in _DRY_RUN_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(sorted(_DRY_RUN_EXTENSIONS))}",
        )
    if ext == ".doc":
        raise HTTPException(
            status_code=400,
            detail=(
                "Файлы .doc (старый формат Word 97-2003) не поддерживаются. "
                "Конвертируйте файл в .docx и повторите попытку."
            ),
        )
    data = await file.read()
    if len(data) > diag_service._max_bytes:
        raise HTTPException(
            status_code=FILE_TOO_LARGE_STATUS,
            detail="File too large for dry-run (max 50 MB)",
        )
    return data


def _compute_quality_warning(page_results: list, types_count: dict, warning_prefix: str) -> str | None:
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


def _build_dry_run_response(
    filename: str,
    page_results: list,
    types_count: dict,
    total_chars: int,
    warning: str | None,
    *,
    preview_id: str | None = None,
    suggestion: str | None = None,
) -> DryRunResponse:
    """Build DryRunResponse from page analysis results."""
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
                image_available=IMAGE_AVAILABLE and p.unit_kind == "page",
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
# Quality list & diagnosis
# ---------------------------------------------------------------------------


@router.get("/admin/documents/quality", response_model=DocumentQualityListResponse)
async def list_quality_documents(
    admin: dict = Depends(require_admin),
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


@router.post("/admin/documents/{document_id}/diagnose", response_model=DocumentDiagnoseResponse)
async def diagnose_document(
    document_id: int,
    admin: dict = Depends(require_admin),
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


@router.post("/admin/documents/preview", response_model=DryRunResponse)
async def dry_run_preview(
    file: UploadFile = File(...),
    admin: dict = Depends(require_admin),
    diag_service: PDFDiagnosticService = Depends(create_pdf_diagnostic_service),
    preview_cache=Depends(create_preview_cache),
):
    """Phase 1: Fast dry-run — text layer only, no OCR."""
    data = await _validate_upload(file, diag_service)

    filename = file.filename or "unnamed"
    ext = Path(filename).suffix.lower()

    try:
        strategy = PreviewStrategyFactory.for_extension(ext, diag_service=diag_service)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    preview_id = await preview_cache.store(data, suffix=ext, original_filename=filename)

    async with preview_cache.get_path(preview_id) as tmp_path:
        if tmp_path is None:
            raise HTTPException(status_code=500, detail="Failed to store preview")
        page_results, types_count, total_chars = strategy.analyze(tmp_path)
        warning = _compute_quality_warning(page_results, types_count, "Low quality")
        suggestion = PDFDiagnosticService.suggest_action(page_results, types_count)
        return _build_dry_run_response(
            filename,
            page_results,
            types_count,
            total_chars,
            warning,
            preview_id=preview_id,
            suggestion=suggestion,
        )


# ---------------------------------------------------------------------------
# Dry-run preview — Phase 2 (OCR)
# ---------------------------------------------------------------------------


@router.post("/admin/documents/preview-ocr", response_model=DryRunResponse)
async def dry_run_ocr_phase2(
    file: UploadFile = File(None),
    preview_id: str = Form(""),
    pages: str = Form(""),
    admin: dict = Depends(require_admin),
    diag_service: PDFDiagnosticService = Depends(create_pdf_diagnostic_service),
    preview_cache=Depends(create_preview_cache),
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
            strategy = PreviewStrategyFactory.for_extension(ext, diag_service=diag_service)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        page_results, types_count, total_chars = strategy.analyze(tmp_path)
        try:
            page_results, types_count, total_chars = strategy.ocr_problem_units(
                tmp_path, page_results, page_nums
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        warning = _compute_quality_warning(page_results, types_count, "Still low quality after OCR")
        suggestion = PDFDiagnosticService.suggest_action(page_results, types_count)
        return _build_dry_run_response(
            fname,
            page_results,
            types_count,
            total_chars,
            warning,
            preview_id=effective_preview_id,
            suggestion=suggestion,
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
    data = await _validate_upload(file, diag_service)
    fname = file.filename or "unnamed"
    ext = Path(fname).suffix.lower()
    new_preview_id = await preview_cache.store(data, suffix=ext, original_filename=fname)

    async with preview_cache.get_path(new_preview_id) as tmp_path:
        if tmp_path is None:
            raise HTTPException(status_code=500, detail="Failed to store preview")
        return await _run_ocr(tmp_path, new_preview_id, fname)


# ---------------------------------------------------------------------------
# Page image rendering (PDF only)
# ---------------------------------------------------------------------------


@router.post("/admin/documents/preview/page-image", response_model=PageImageResponse)
async def get_page_image(
    preview_id: str = Form(...),
    page: int = Form(...),
    admin: dict = Depends(require_admin),
    diag_service: PDFDiagnosticService = Depends(create_pdf_diagnostic_service),
    preview_cache=Depends(create_preview_cache),
):
    """Render a single page of the cached PDF as a PNG image (base64-encoded)."""
    if not IMAGE_AVAILABLE:
        raise HTTPException(
            status_code=404,
            detail="Page image rendering is not available (requires S3 storage backend)",
        )

    async with preview_cache.get_path(preview_id) as tmp_path:
        if tmp_path is None:
            raise HTTPException(status_code=404, detail="Preview expired or not found")

        if page < 1:
            raise HTTPException(status_code=400, detail="Page number must be >= 1")

        doc = diag_service._pdf.open(str(tmp_path))
        try:
            total = diag_service._pdf.get_page_count(doc)
            if page > total:
                raise HTTPException(
                    status_code=400,
                    detail=f"Page {page} out of range (document has {total} pages)",
                )
            image_bytes = diag_service._pdf.render_page_image(doc, page - 1, dpi=PAGE_IMAGE_DPI)
        finally:
            diag_service._pdf.close(doc)

        return PageImageResponse(
            image_base64=base64.b64encode(image_bytes).decode("ascii"),
            page=page,
        )


# ---------------------------------------------------------------------------
# Index directly from dry-run
# ---------------------------------------------------------------------------


@router.post("/admin/documents/preview/{preview_id}/index")
async def index_from_preview(
    preview_id: str,
    visibility: str = Form("internal_public"),
    group_id: int | None = Form(None),
    client_id: int | None = Form(None),
    doc_domain: str | None = Form(None),
    admin: dict = Depends(require_admin),
    preview_cache=Depends(create_preview_cache),
    document_service=Depends(create_document_service),
    job_service=Depends(create_job_service),
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
        user_id=admin["id"],
        user_kind=admin["kind"],
        user_role=admin["role"],
        rename_on_conflict=False,
        doc_domain=doc_domain,
        document_service=document_service,
        job_service=job_service,
        enqueue_fn=enqueue_document_processing,
        action_name="document.upload_from_preview",
    )

    return {"document_id": result["document_id"], "filename": filename, "status": result["status"]}
