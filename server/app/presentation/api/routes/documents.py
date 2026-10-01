"""Document endpoints — thin wrappers around DocumentCommandService/QueryService."""

from __future__ import annotations

import logging
import hashlib

from application.ports.rate_limit import RateLimitPolicyName
from application.services.document_command_service import DocumentCommandService
from application.services.document_query_service import DocumentQueryService
from application.services.job_service import JobService
from domain.value_objects.capabilities import Capability
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.visibility import DocumentVisibility
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile

from presentation.api.auth_dependencies import get_current_user, require_capability
from presentation.api.rate_limit import rate_limit
from presentation.api.dependencies import (
    create_action_logger,
    create_cache_invalidator,
    create_cache_config,
    create_document_command_service,
    create_document_query_service,
    create_idempotency_store,
    create_job_enqueuer,
    create_job_service,
    create_upload_config,
    get_idempotency_key,
)
from presentation.api.helpers import read_upload_with_limit, upload_and_enqueue
from presentation.api.idempotency import complete_idempotency, reserve_idempotency
from presentation.api.schemas.validators import validate_uploaded_file
from presentation.api.schemas import (
    CurrentUser,
    DeleteDocumentResponse,
    DocumentRenameRequest,
    DocumentResponse,
    UploadStatusResponse,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["documents"])


@router.get("/documents/clients")
async def list_uploadable_clients(
    current_user: CurrentUser = Depends(get_current_user),
    query: DocumentQueryService = Depends(create_document_query_service),
):
    """List clients available for client_private upload (assigned clients for internal, self for client)."""
    return await query.list_uploadable_clients(current_user.id, current_user.kind, current_user.role)


@router.post(
    "/documents",
    response_model=UploadStatusResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.UPLOAD))],
)
async def upload_document(
    request: Request,
    current_user: CurrentUser = Depends(require_capability(Capability.DOCUMENTS_MANAGE)),
    file: UploadFile = File(...),
    visibility: DocumentVisibility = Form(...),
    group_id: int | None = Form(None),
    client_id: int | None = Form(None),
    rename_on_conflict: bool = Form(False),
    doc_domain: str | None = Form(None),
    replaces_document_id: int | None = Form(None),
    cmd: DocumentCommandService = Depends(create_document_command_service),
    job_service: JobService = Depends(create_job_service),
    job_enqueuer=Depends(create_job_enqueuer),
    log=Depends(create_action_logger),
    upload_cfg=Depends(create_upload_config),
    idempotency_key: str | None = Depends(get_idempotency_key),
    idempotency_store=Depends(create_idempotency_store),
):
    filename = file.filename or "unnamed"

    if doc_domain is not None and doc_domain not in [d.value for d in DocDomain]:
        raise HTTPException(status_code=400, detail="doc_domain must be 'legal' or 'general'")

    max_bytes = upload_cfg.max_upload_size_mb * 1024 * 1024
    data = await read_upload_with_limit(file, max_bytes)

    validate_uploaded_file(data, filename)

    cached, reservation = await reserve_idempotency(
        idempotency_store,
        idempotency_key,
        f"{current_user.kind}:{current_user.role}:{current_user.id}",
        f"{request.method}:{request.url.path}",
        {
            "filename": filename,
            "file_sha256": hashlib.sha256(data).hexdigest(),
            "visibility": visibility.value,
            "group_id": group_id,
            "client_id": client_id,
            "rename_on_conflict": rename_on_conflict,
            "doc_domain": doc_domain,
            "replaces_document_id": replaces_document_id,
        },
    )
    if cached is not None:
        return UploadStatusResponse(**cached)

    result = await upload_and_enqueue(
        file_data=data,
        filename=filename,
        visibility=visibility.value,
        group_id=group_id,
        client_id=client_id,
        user_id=current_user.id,
        user_kind=current_user.kind,
        user_role=current_user.role,
        rename_on_conflict=rename_on_conflict,
        doc_domain=doc_domain,
        replaces_document_id=replaces_document_id,
        document_service=cmd,
        job_service=job_service,
        enqueue_fn=job_enqueuer.enqueue_document_processing,
        action_name="document.upload",
        log_fn=log,
    )

    response = UploadStatusResponse(
        status=result["status"], document_id=result["document_id"], filename=filename
    )

    # Idempotency: store result for future duplicate requests
    await complete_idempotency(idempotency_store, reservation, response.model_dump(mode="json"))

    return response


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    limit: int = Query(200, ge=1, le=1000, description="Page size (M-6: unbounded lists are forbidden)"),
    offset: int = Query(0, ge=0),
    current_user: CurrentUser = Depends(require_capability(Capability.DOCUMENTS_VIEW)),
    query: DocumentQueryService = Depends(create_document_query_service),
):
    return await query.list_documents(
        user_id=current_user.id,
        user_kind=current_user.kind,
        user_role=current_user.role,
        limit=limit,
        offset=offset,
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document_status(
    document_id: int,
    current_user: CurrentUser = Depends(require_capability(Capability.DOCUMENTS_VIEW)),
    query: DocumentQueryService = Depends(create_document_query_service),
):
    return await query.get_document(document_id, current_user.id, current_user.kind, current_user.role)


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def delete_document(
    document_id: int,
    current_user: CurrentUser = Depends(require_capability(Capability.DOCUMENTS_MANAGE)),
    cmd: DocumentCommandService = Depends(create_document_command_service),
    log=Depends(create_action_logger),
    cache_inv=Depends(create_cache_invalidator),
    cache_cfg=Depends(create_cache_config),
):
    await cmd.delete_document(document_id, current_user.id, current_user.role, current_user.kind)
    await cache_inv.invalidate_by_document_ids([document_id], cache_enabled=cache_cfg.cache_enabled)
    log("document.delete", user_id=current_user.id, details={"document_id": document_id})
    return DeleteDocumentResponse(status="deleted", document_id=document_id)


@router.patch(
    "/documents/{document_id}/rename",
    response_model=DocumentResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def rename_document(
    document_id: int,
    body: DocumentRenameRequest,
    current_user: CurrentUser = Depends(require_capability(Capability.DOCUMENTS_MANAGE)),
    cmd: DocumentCommandService = Depends(create_document_command_service),
    log=Depends(create_action_logger),
):
    result = await cmd.rename_document(
        document_id=document_id,
        new_filename=body.filename,
        user_id=current_user.id,
        user_role=current_user.role,
        user_kind=current_user.kind,
    )
    log(
        "document.rename",
        user_id=current_user.id,
        details={"document_id": document_id, "new_filename": body.filename},
    )
    return result
