"""Admin quality / diagnostics endpoints — PDF extraction quality, dry-run preview."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import fitz
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from infrastructure.ml.ingestion import clean_pdf_text, ocr_pdf_pages
from infrastructure.ml.pdf_diag import classify_page

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import get_uow_factory
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


@router.get("/admin/documents/quality", response_model=DocumentQualityListResponse)
async def list_quality_documents(admin: dict = Depends(require_admin)):
    """List documents with quality warnings, sorted by quality_score descending."""
    uow_factory = get_uow_factory()
    async with uow_factory.create() as uow:
        all_docs = await uow.documents.list_all()
        warned = [
            d
            for d in all_docs
            if d.warning_message or (d.quality_score is not None and d.quality_score > 0.3)
        ]
        warned.sort(key=lambda d: d.quality_score or 0.0, reverse=True)

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
):
    """Run per-page PDF diagnosis on an already-indexed document.

    Returns page-by-page classification (text/scan/garbled/empty/table).
    """
    uow_factory = get_uow_factory()
    async with uow_factory.create() as uow:
        doc = await uow.documents.get_by_id(document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")

        if not doc.source_path:
            raise HTTPException(status_code=400, detail="Document has no source file")

        from infrastructure.storage import LazyStorage

        storage = LazyStorage()
        temp_path = storage.download_to_temp(doc.source_path)

        try:
            pdf = fitz.open(str(temp_path))
            total_pages = len(pdf)
            page_diagnostics = []

            for i, page in enumerate(pdf):
                text = page.get_text("text")
                chars = len(text.strip())
                ptype, desc = classify_page(text, chars, page=page)
                page_diagnostics.append(
                    PageDiagnostic(
                        page=i + 1,
                        type=ptype,
                        chars=chars,
                        description=desc,
                    )
                )

            pdf.close()

            types = [p.type for p in page_diagnostics]
            summary = {
                "text": types.count("text"),
                "scan": types.count("scan"),
                "garbled": types.count("garbled"),
                "empty": types.count("empty"),
                "table": types.count("table"),
            }

            return DocumentDiagnoseResponse(
                document_id=document_id,
                filename=doc.filename,
                total_pages=total_pages,
                pages=page_diagnostics,
                summary=summary,
            )
        finally:
            temp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Dry-run: two-phase approach
# ---------------------------------------------------------------------------


def _analyze_text_layer(pdf_path: Path) -> tuple[list[DryRunPageResult], dict[str, int], int]:
    """Phase 1: Fast text-layer-only analysis. No OCR.

    Returns (page_results, types_count, total_chars).
    """
    pdf = fitz.open(str(pdf_path))
    page_results: list[DryRunPageResult] = []
    types_count: dict[str, int] = {"text": 0, "scan": 0, "garbled": 0, "empty": 0, "table": 0}
    total_chars = 0

    for i in range(len(pdf)):
        page_num = i + 1
        page = pdf.load_page(i)

        # Text layer
        raw_text = page.get_text("text")
        chars = len(raw_text.strip())

        # Table detection
        has_table = False
        try:
            tables = page.find_tables()
            has_table = bool(tables and tables.tables)
        except Exception:
            pass

        # Classify
        if has_table:
            ptype = "table"
        else:
            ptype, _desc = classify_page(raw_text, chars, page=page)

        types_count[ptype] = types_count.get(ptype, 0) + 1

        cleaned = clean_pdf_text(raw_text) if raw_text else ""
        total_chars += len(cleaned)

        page_results.append(
            DryRunPageResult(
                page=page_num,
                type=ptype,
                content_type="table" if has_table else "text",
                chars=chars,
                preview=cleaned[:200] if cleaned else raw_text[:200],
            )
        )

    pdf.close()
    return page_results, types_count, total_chars


def _ocr_problem_pages(
    pdf_path: Path,
    page_results: list[DryRunPageResult],
) -> tuple[list[DryRunPageResult], dict[str, int], int]:
    """Phase 2: Run OCR only on scan/empty pages, merge results."""

    # Pages that need OCR: scan or empty
    problem_pages = [p.page for p in page_results if p.type in ("scan", "empty")]
    if not problem_pages:
        return page_results, {}, 0

    pdf = fitz.open(str(pdf_path))
    ocr_results = ocr_pdf_pages(pdf, problem_pages, pdf_path.name)
    pdf.close()

    # Update page results
    new_types: dict[str, int] = {"text": 0, "scan": 0, "garbled": 0, "empty": 0, "table": 0}
    total_chars = 0

    for pr in page_results:
        if pr.page in ocr_results:
            ocr_text = ocr_results[pr.page]
            if ocr_text:
                ocr_text = clean_pdf_text(ocr_text)
                if ocr_text:
                    pr.type = "text"
                    pr.content_type = "ocr"
                    pr.chars = len(ocr_text)
                    pr.preview = ocr_text[:200]
                else:
                    # OCR ran but produced nothing useful
                    pr.type = "scan"
                    pr.content_type = "ocr"
            else:
                pr.type = "scan"
                pr.content_type = "ocr"

        new_types[pr.type] = new_types.get(pr.type, 0) + 1
        total_chars += pr.chars

    return page_results, new_types, total_chars


@router.post("/admin/documents/preview", response_model=DryRunResponse)
async def dry_run_preview(
    file: UploadFile = File(...),
    admin: dict = Depends(require_admin),
):
    """Phase 1: Fast dry-run — text layer only, no OCR.

    Analyzes PDF text layer instantly. Returns page classifications,
    heatmap data, and anomaly list. OCR is NOT run.
    """
    filename = file.filename or "unnamed"
    ext = Path(filename).suffix.lower()

    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Dry-run currently only supports PDF files")

    data = await file.read()
    max_bytes = 50 * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large for dry-run (max 50 MB)")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        page_results, types_count, total_chars = _analyze_text_layer(tmp_path)
        total_pages = len(page_results)

        # Detect low quality
        bad = types_count.get("scan", 0) + types_count.get("garbled", 0) + types_count.get("empty", 0)
        bad_ratio = bad / total_pages if total_pages else 0.0
        warning = None
        if bad_ratio > 0.3:
            warning = (
                f"Low quality: {types_count.get('scan', 0)} scan + "
                f"{types_count.get('garbled', 0)} garbled + "
                f"{types_count.get('empty', 0)} empty "
                f"out of {total_pages} pages ({bad_ratio:.0%} bad)"
            )

        # Build a sample of full text from text-type pages for preview
        full_text_preview = "\n\n".join(p.preview for p in page_results if p.type == "text")[:2000]

        return DryRunResponse(
            filename=filename,
            total_pages=total_pages,
            pages=page_results,
            total_chars=total_chars,
            quality_score=bad_ratio,
            warning=warning,
            full_text_preview=full_text_preview,
            summary=types_count,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/admin/documents/preview-ocr", response_model=DryRunResponse)
async def dry_run_ocr_phase2(
    file: UploadFile = File(...),
    pages: str = Form(""),
    admin: dict = Depends(require_admin),
):
    """Phase 2: Run OCR on specific problem pages and return updated results.

    Accepts the same PDF file + comma-separated page numbers to OCR.
    Only the specified pages get OCR — everything else is fast text-layer.
    """
    filename = file.filename or "unnamed"
    ext = Path(filename).suffix.lower()

    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Dry-run currently only supports PDF files")

    # Parse page numbers
    try:
        page_nums = [int(p.strip()) for p in pages.split(",") if p.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid page numbers")

    if not page_nums:
        raise HTTPException(status_code=400, detail="No pages specified for OCR")

    data = await file.read()
    max_bytes = 50 * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large for dry-run (max 50 MB)")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        # Phase 1: fast text layer for ALL pages
        page_results, _types_count, _total_chars = _analyze_text_layer(tmp_path)

        # Phase 2: OCR only specified pages
        pdf = fitz.open(str(tmp_path))
        ocr_results = ocr_pdf_pages(pdf, page_nums, filename)
        pdf.close()

        # Merge OCR results into page_results
        for pr in page_results:
            if pr.page in ocr_results:
                ocr_text = ocr_results[pr.page]
                if ocr_text:
                    ocr_text = clean_pdf_text(ocr_text)
                    if ocr_text:
                        pr.type = "text"
                        pr.content_type = "ocr"
                        pr.chars = len(ocr_text)
                        pr.preview = ocr_text[:200]
                    else:
                        pr.type = "scan"
                        pr.content_type = "ocr"
                else:
                    pr.type = "scan"
                    pr.content_type = "ocr"

        # Recompute summary
        types_count: dict[str, int] = {"text": 0, "scan": 0, "garbled": 0, "empty": 0, "table": 0}
        total_chars = 0
        for pr in page_results:
            types_count[pr.type] = types_count.get(pr.type, 0) + 1
            total_chars += pr.chars

        total_pages = len(page_results)
        bad = types_count.get("scan", 0) + types_count.get("garbled", 0) + types_count.get("empty", 0)
        bad_ratio = bad / total_pages if total_pages else 0.0
        warning = None
        if bad_ratio > 0.3:
            warning = (
                f"Still low quality after OCR: {types_count.get('scan', 0)} scan + "
                f"{types_count.get('garbled', 0)} garbled + "
                f"{types_count.get('empty', 0)} empty "
                f"out of {total_pages} pages ({bad_ratio:.0%} bad)"
            )

        full_text_preview = "\n\n".join(p.preview for p in page_results if p.type == "text")[:2000]

        return DryRunResponse(
            filename=filename,
            total_pages=total_pages,
            pages=page_results,
            total_chars=total_chars,
            quality_score=bad_ratio,
            warning=warning,
            full_text_preview=full_text_preview,
            summary=types_count,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
