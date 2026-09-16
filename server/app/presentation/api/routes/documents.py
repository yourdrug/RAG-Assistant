"""Document endpoints — thin wrappers around DocumentService."""

from __future__ import annotations

import logging

from application.services.document_service import DocumentService
from application.services.job_service import JobService
from domain.value_objects.capabilities import Capability
from domain.value_objects.doc_domain import DocDomain
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from presentation.api.auth_dependencies import get_current_user, require_capability
from presentation.api.constants import FILE_TOO_LARGE_STATUS
from presentation.api.dependencies import (
    create_action_logger,
    create_cache_invalidator,
    create_cache_config,
    create_document_service,
    create_job_enqueuer,
    create_job_service,
    create_upload_config,
)
from presentation.api.helpers import upload_and_enqueue
from presentation.api.schemas import (
    CurrentUser,
    DocumentRenameRequest,
    DocumentResponse,
    FileContent,
    UploadStatusResponse,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["documents"])


@router.get("/documents/clients")
async def list_uploadable_clients(
    current_user: CurrentUser = Depends(get_current_user),
    document_service: DocumentService = Depends(create_document_service),
):
    """List clients available for client_private upload (assigned clients for internal, self for client)."""
    return await document_service.list_uploadable_clients(
        current_user.id, current_user.kind, current_user.role
    )


@router.post("/documents", response_model=UploadStatusResponse)
async def upload_document(
    current_user: CurrentUser = Depends(require_capability(Capability.DOCUMENTS_MANAGE)),
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
    upload_cfg=Depends(create_upload_config),
):
    filename = file.filename or "unnamed"

    if doc_domain is not None and doc_domain not in [d.value for d in DocDomain]:
        raise HTTPException(status_code=400, detail="doc_domain must be 'legal' or 'general'")

    max_bytes = upload_cfg.max_upload_size_mb * 1024 * 1024
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=FILE_TOO_LARGE_STATUS,
            detail=(
                f"File too large: {len(data) / 1024 / 1024:.1f} MB (limit {upload_cfg.max_upload_size_mb} MB)"
            ),
        )

    FileContent(data=data, filename=filename, check_structural=True)

    result = await upload_and_enqueue(
        file_data=data,
        filename=filename,
        visibility=visibility,
        group_id=group_id,
        client_id=client_id,
        user_id=current_user.id,
        user_kind=current_user.kind,
        user_role=current_user.role,
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
    current_user: CurrentUser = Depends(require_capability(Capability.DOCUMENTS_VIEW)),
    document_service: DocumentService = Depends(create_document_service),
):
    return await document_service.list_documents(
        current_user.id, current_user.kind, current_user.role, limit=limit, offset=offset
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document_status(
    document_id: int,
    current_user: CurrentUser = Depends(require_capability(Capability.DOCUMENTS_VIEW)),
    document_service: DocumentService = Depends(create_document_service),
):
    return await document_service.get_document(
        document_id, current_user.id, current_user.kind, current_user.role
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: CurrentUser = Depends(require_capability(Capability.DOCUMENTS_MANAGE)),
    document_service: DocumentService = Depends(create_document_service),
    log=Depends(create_action_logger),
    cache_inv=Depends(create_cache_invalidator),
    cache_cfg=Depends(create_cache_config),
):
    await document_service.delete_document(document_id, current_user.id, current_user.role)
    await cache_inv.invalidate_by_document_ids([document_id], cache_enabled=cache_cfg.cache_enabled)
    log("document.delete", user_id=current_user.id, details={"document_id": document_id})
    return {"status": "deleted", "document_id": document_id}


@router.patch("/documents/{document_id}/rename", response_model=DocumentResponse)
async def rename_document(
    document_id: int,
    body: DocumentRenameRequest,
    current_user: CurrentUser = Depends(require_capability(Capability.DOCUMENTS_MANAGE)),
    document_service: DocumentService = Depends(create_document_service),
    log=Depends(create_action_logger),
):
    result = await document_service.rename_document(
        document_id=document_id,
        new_filename=body.filename,
        user_id=current_user.id,
        user_role=current_user.role,
    )
    log(
        "document.rename",
        user_id=current_user.id,
        details={"document_id": document_id, "new_filename": body.filename},
    )
    return result
