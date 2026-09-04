"""Document endpoints — thin wrappers around DocumentService."""

from __future__ import annotations

import logging
from pathlib import Path

from application.services.document_service import DocumentService
from application.services.job_service import JobService
from config import settings
from domain.value_objects.doc_domain import DocDomain
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from infrastructure.logging.actions import log_action
from infrastructure.worker.queue import enqueue_document_processing

from presentation.api.auth_dependencies import get_current_user
from presentation.api.constants import FILE_TOO_LARGE_STATUS, MAGIC_BYTES
from presentation.api.dependencies import (
    create_document_service,
    create_job_service,
)
from presentation.api.helpers import upload_and_enqueue
from presentation.api.rate_limits import upload_rate_limit
from presentation.api.schemas import DocumentRenameRequest, DocumentResponse, UploadStatusResponse

logger = logging.getLogger("default")

router = APIRouter(tags=["documents"])


def _validate_mime(file_data: bytes, extension: str) -> None:
    """Verify file contents match declared extension using magic bytes."""
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


@router.get("/documents/clients")
async def list_uploadable_clients(
    current_user: dict = Depends(get_current_user),
    document_service: DocumentService = Depends(create_document_service),
):
    """List clients available for client_private upload (assigned clients for internal, self for client)."""
    return await document_service.list_uploadable_clients(
        current_user["id"], current_user["kind"], current_user["role"]
    )


@router.post("/documents", response_model=UploadStatusResponse, dependencies=[Depends(upload_rate_limit)])
async def upload_document(
    current_user: dict = Depends(get_current_user),
    file: UploadFile = File(...),
    visibility: str = Form(...),
    group_id: int | None = Form(None),
    client_id: int | None = Form(None),
    rename_on_conflict: bool = Form(False),
    doc_domain: str | None = Form(None),
    document_service: DocumentService = Depends(create_document_service),
    job_service: JobService = Depends(create_job_service),
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
                f"File too large: {len(data) / 1024 / 1024:.1f} MB"
                f" (limit {settings.max_upload_size_mb} MB)"
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
        document_service=document_service,
        job_service=job_service,
        enqueue_fn=enqueue_document_processing,
        action_name="document.upload",
    )

    return UploadStatusResponse(status=result["status"], document_id=result["document_id"], filename=filename)


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    current_user: dict = Depends(get_current_user),
    document_service: DocumentService = Depends(create_document_service),
):
    return await document_service.list_documents(
        current_user["id"], current_user["kind"], current_user["role"]
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document_status(
    document_id: int,
    current_user: dict = Depends(get_current_user),
    document_service: DocumentService = Depends(create_document_service),
):
    return await document_service.get_document(
        document_id, current_user["id"], current_user["kind"], current_user["role"]
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: dict = Depends(get_current_user),
    document_service: DocumentService = Depends(create_document_service),
):
    await document_service.delete_document(document_id, current_user["id"], current_user["role"])
    log_action("document.delete", user_id=current_user["id"], details={"document_id": document_id})
    return {"status": "deleted", "document_id": document_id}


@router.patch("/documents/{document_id}/rename", response_model=DocumentResponse)
async def rename_document(
    document_id: int,
    body: DocumentRenameRequest,
    current_user: dict = Depends(get_current_user),
    document_service: DocumentService = Depends(create_document_service),
):
    result = await document_service.rename_document(
        document_id=document_id,
        new_filename=body.filename,
        user_id=current_user["id"],
        user_role=current_user["role"],
    )
    log_action(
        "document.rename",
        user_id=current_user["id"],
        details={"document_id": document_id, "new_filename": body.filename},
    )
    return result
