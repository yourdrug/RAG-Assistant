"""Admin quality / diagnostics endpoints — PDF extraction quality, dry-run preview."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from application.services.pdf_diagnostic_service import PDFDiagnosticService
from application.services.quality_service import QualityService
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import create_pdf_diagnostic_service, create_quality_service
from presentation.api.schemas import (
    DocumentDiagnoseResponse,
    DocumentQualityItem,
    DocumentQualityListResponse,
    DryRunPageResult,
    DryRunResponse,
    PageDiagnostic,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["admin-quality"])


async def _validate_pdf_upload(file: UploadFile, diag_service: PDFDiagnosticService) -> bytes:
    """Validate uploaded file is PDF and within size limits. Returns file data."""
    filename = file.filename or "unnamed"
    ext = Path(filename).suffix.lower()
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Dry-run currently only supports PDF files")
    data = await file.read()
    if len(data) > diag_service._max_bytes:
        raise HTTPException(status_code=413, detail="File too large for dry-run (max 50 MB)")
    return data


def _compute_quality_warning(page_results: list, types_count: dict, warning_prefix: str) -> str | None:
    """Compute quality warning from page analysis results."""
    total_pages = len(page_results)
    bad = types_count.get("scan", 0) + types_count.get("garbled", 0) + types_count.get("empty", 0)
    bad_ratio = bad / total_pages if total_pages else 0.0
    if bad_ratio > 0.3:
        return (
            f"{warning_prefix}: {types_count.get('scan', 0)} scan + "
            f"{types_count.get('garbled', 0)} garbled + "
            f"{types_count.get('empty', 0)} empty "
            f"out of {total_pages} pages ({bad_ratio:.0%} bad)"
        )
    return None


def _build_dry_run_response(
    filename: str,
    page_results: list,
    types_count: dict,
    total_chars: int,
    warning: str | None,
) -> DryRunResponse:
    """Build DryRunResponse from page analysis results."""
    total_pages = len(page_results)
    bad = types_count.get("scan", 0) + types_count.get("garbled", 0) + types_count.get("empty", 0)
    bad_ratio = bad / total_pages if total_pages else 0.0
    full_text_preview = "\n\n".join(p.preview for p in page_results if p.type == "text")[:2000]
    return DryRunResponse(
        filename=filename,
        total_pages=total_pages,
        pages=[
            DryRunPageResult(
                page=p.page, type=p.type, content_type=p.content_type, chars=p.chars, preview=p.preview
            )
            for p in page_results
        ],
        total_chars=total_chars,
        quality_score=bad_ratio,
        warning=warning,
        full_text_preview=full_text_preview,
        summary=types_count,
    )


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


@router.post("/admin/documents/preview", response_model=DryRunResponse)
async def dry_run_preview(
    file: UploadFile = File(...),
    admin: dict = Depends(require_admin),
    diag_service: PDFDiagnosticService = Depends(create_pdf_diagnostic_service),
):
    """Phase 1: Fast dry-run — text layer only, no OCR."""
    data = await _validate_pdf_upload(file, diag_service)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        page_results, types_count, total_chars = diag_service.analyze_text_layer(tmp_path)
        warning = _compute_quality_warning(page_results, types_count, "Low quality")
        return _build_dry_run_response(
            file.filename or "unnamed", page_results, types_count, total_chars, warning
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/admin/documents/preview-ocr", response_model=DryRunResponse)
async def dry_run_ocr_phase2(
    file: UploadFile = File(...),
    pages: str = Form(""),
    admin: dict = Depends(require_admin),
    diag_service: PDFDiagnosticService = Depends(create_pdf_diagnostic_service),
):
    """Phase 2: Run OCR on specific problem pages and return updated results."""
    try:
        page_nums = [int(p.strip()) for p in pages.split(",") if p.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid page numbers") from None

    if not page_nums:
        raise HTTPException(status_code=400, detail="No pages specified for OCR")

    data = await _validate_pdf_upload(file, diag_service)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        page_results, _types_count, _total_chars = diag_service.analyze_text_layer(tmp_path)
        page_results, types_count, total_chars = diag_service.ocr_problem_pages(tmp_path, page_results)
        warning = _compute_quality_warning(page_results, types_count, "Still low quality after OCR")
        return _build_dry_run_response(
            file.filename or "unnamed", page_results, types_count, total_chars, warning
        )
    finally:
        tmp_path.unlink(missing_ok=True)
