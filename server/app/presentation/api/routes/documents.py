"""Document endpoints — thin wrappers around DocumentService."""

from __future__ import annotations

import logging
from pathlib import Path

from application.services.document_service import DocumentService
from application.services.job_service import JobService
from config import settings
from domain.value_objects.capabilities import Capability
from domain.value_objects.doc_domain import DocDomain
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from presentation.api.auth_dependencies import get_current_user, require_capability
from presentation.api.constants import FILE_TOO_LARGE_STATUS, MAGIC_BYTES
from presentation.api.dependencies import (
    create_action_logger,
    create_cache_invalidator,
    create_document_service,
    create_job_enqueuer,
    create_job_service,
)
from presentation.api.helpers import upload_and_enqueue
from presentation.api.schemas import DocumentRenameRequest, DocumentResponse, UploadStatusResponse

logger = logging.getLogger("default")

router = APIRouter(tags=["documents"])


def _validate_mime(file_data: bytes, extension: str) -> None:
    """Verify file contents match declared extension using magic bytes + structural checks.

    Beyond the leading signature, container formats that can be cheaply
    sanity-checked (ZIP-based .docx, PDF) are validated structurally to reject
    trivial signature-spoofed files (polyglots). This is not a substitute for
    antivirus, but it raises the bar for accidental/casual bypass.
    """
    if extension not in MAGIC_BYTES:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {extension}")
    expected = MAGIC_BYTES[extension]
    if not expected:
        return  # plain text — no magic to check
    if not any(file_data[: len(sig)] == sig for sig in expected):
        raise HTTPException(
            status_code=400,
            detail=f"File content does not match extension {extension}",
        )
    if extension == ".pdf" and not _is_valid_pdf(file_data):
        raise HTTPException(
            status_code=400,
            detail="File content is not a valid PDF (missing EOF marker)",
        )
    if extension == ".docx" and not _is_zip_archive(file_data):
        raise HTTPException(
            status_code=400,
            detail="File content is not a valid ZIP-archive (.docx)",
        )
    if extension == ".doc" and not _is_ole2_document(file_data):
        raise HTTPException(
            status_code=400,
            detail="File content is not a valid OLE2 compound document (.doc)",
        )


def _is_zip_archive(data: bytes) -> bool:
    """Cheap check that data contains a ZIP End Of Central Directory record.

    The EOCD signature marks the structural end of any ZIP archive, which a
    signature-only spoof would not contain.
    """
    if len(data) < 22:
        return False
    return b"PK\x05\x06" in data[-65557:]


def _is_valid_pdf(data: bytes) -> bool:
    """Check that PDF ends with %%EOF marker (standard PDF trailer)."""
    tail = data[-1024:] if len(data) > 1024 else data
    return b"%%EOF" in tail


def _is_ole2_document(data: bytes) -> bool:
    """Verify OLE2 compound document beyond magic bytes.

    Checks for the CFB (Compound File Binary) header structure:
    magic + version + byte order + sector size.
    """
    if len(data) < 512:
        return False
    # Bytes 26-27: minor version (should be 0x003E for OLE2)
    # Bytes 28-29: major version (0x0003 or 0x0004)
    # Bytes 30-31: byte order (0xFFFE = little-endian)
    # Bytes 32-33: sector size power (9 = 512 bytes)
    return data[26:28] == b"\x3e\x00" and data[30:32] == b"\xfe\xff" and data[32] == 0x09


@router.get("/documents/clients")
async def list_uploadable_clients(
    current_user: dict = Depends(get_current_user),
    document_service: DocumentService = Depends(create_document_service),
):
    """List clients available for client_private upload (assigned clients for internal, self for client)."""
    return await document_service.list_uploadable_clients(
        current_user["id"], current_user["kind"], current_user["role"]
    )


@router.post("/documents", response_model=UploadStatusResponse)
async def upload_document(
    current_user: dict = Depends(require_capability(Capability.DOCUMENTS_MANAGE)),
    file: UploadFile = File(...),
    visibility: str = Form(...),
    group_id: int | None = Form(None),
    client_id: int | None = Form(None),
    rename_on_conflict: bool = Form(False),
    doc_domain: str | None = Form(None),
    replaces_document_id: int | None = Form(None),
    document_service: DocumentService = Depends(create_document_service),
    job_service: JobService = Depends(create_job_service),
    job_enqueuer=Depends(create_job_enqueuer),
    log=Depends(create_action_logger),
):
    filename = file.filename or "unnamed"
    ext = Path(filename).suffix.lower()

    if doc_domain is not None and doc_domain not in [d.value for d in DocDomain]:
        raise HTTPException(status_code=400, detail="doc_domain must be 'legal' or 'general'")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=FILE_TOO_LARGE_STATUS,
            detail=(
                f"File too large: {len(data) / 1024 / 1024:.1f} MB (limit {settings.max_upload_size_mb} MB)"
            ),
        )

    _validate_mime(data, ext)

    result = await upload_and_enqueue(
        file_data=data,
        filename=filename,
        visibility=visibility,
        group_id=group_id,
        client_id=client_id,
        user_id=current_user["id"],
        user_kind=current_user["kind"],
        user_role=current_user["role"],
        rename_on_conflict=rename_on_conflict,
        doc_domain=doc_domain,
        replaces_document_id=replaces_document_id,
        document_service=document_service,
        job_service=job_service,
        enqueue_fn=job_enqueuer.enqueue_document_processing,
        action_name="document.upload",
        log_fn=log,
    )

    return UploadStatusResponse(status=result["status"], document_id=result["document_id"], filename=filename)


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    limit: int = Query(200, ge=1, le=1000, description="Page size (M-6: unbounded lists are forbidden)"),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(require_capability(Capability.DOCUMENTS_VIEW)),
    document_service: DocumentService = Depends(create_document_service),
):
    return await document_service.list_documents(
        current_user["id"], current_user["kind"], current_user["role"], limit=limit, offset=offset
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document_status(
    document_id: int,
    current_user: dict = Depends(require_capability(Capability.DOCUMENTS_VIEW)),
    document_service: DocumentService = Depends(create_document_service),
):
    return await document_service.get_document(
        document_id, current_user["id"], current_user["kind"], current_user["role"]
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: dict = Depends(require_capability(Capability.DOCUMENTS_MANAGE)),
    document_service: DocumentService = Depends(create_document_service),
    log=Depends(create_action_logger),
    cache_inv=Depends(create_cache_invalidator),
):
    await document_service.delete_document(document_id, current_user["id"], current_user["role"])
    await cache_inv.invalidate_by_document_ids([document_id], cache_enabled=settings.cache_enabled)
    log("document.delete", user_id=current_user["id"], details={"document_id": document_id})
    return {"status": "deleted", "document_id": document_id}


@router.patch("/documents/{document_id}/rename", response_model=DocumentResponse)
async def rename_document(
    document_id: int,
    body: DocumentRenameRequest,
    current_user: dict = Depends(require_capability(Capability.DOCUMENTS_MANAGE)),
    document_service: DocumentService = Depends(create_document_service),
    log=Depends(create_action_logger),
):
    result = await document_service.rename_document(
        document_id=document_id,
        new_filename=body.filename,
        user_id=current_user["id"],
        user_role=current_user["role"],
    )
    log(
        "document.rename",
        user_id=current_user["id"],
        details={"document_id": document_id, "new_filename": body.filename},
    )
    return result
