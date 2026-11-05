"""Document endpoints — thin wrappers around DocumentCommandService/QueryService."""

import asyncio
from dataclasses import dataclass
import hashlib
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile

from application.documents import (
    DocumentCommandService,
    DocumentDeletionJobs,
    DocumentDTO,
    DocumentQueryService,
)
from application.documents.dto import ClientInfo
from application.ports.action_logger import ActionLoggerPort
from application.ports.rate_limit import RateLimitPolicyName
from application.ports.route_configs import UploadConfig
from domain.identity.capabilities import Capability
from domain.knowledge.enums import DocDomain, DocumentVisibility
from domain.runtime.jobs import BackgroundJobStatus
from presentation.api.auth_dependencies import get_current_user, require_capability
from presentation.api.dependencies import (
    IdempotencyContext,
    JobSubmission,
    create_action_logger,
    create_document_command_service,
    create_document_deletion_jobs,
    create_document_query_service,
    create_idempotency_context,
    create_job_submission,
    create_upload_config,
)
from presentation.api.helpers import read_upload_with_limit, upload_and_enqueue
from presentation.api.idempotency import complete_idempotency, reserve_idempotency
from presentation.api.rate_limit import rate_limit
from presentation.api.schemas import (
    CurrentUser,
    DeleteDocumentResponse,
    DocumentRenameRequest,
    DocumentResponse,
    UploadStatusResponse,
)
from presentation.api.schemas.validators import validate_uploaded_file


logger = logging.getLogger("default")


@dataclass(kw_only=True)
class DocumentUploadForm:
    file: Annotated[UploadFile, File()]
    visibility: Annotated[DocumentVisibility, Form()]
    group_id: Annotated[int | None, Form()] = None
    client_id: Annotated[int | None, Form()] = None
    rename_on_conflict: Annotated[bool, Form()] = False
    doc_domain: Annotated[str | None, Form()] = None
    replaces_document_id: Annotated[int | None, Form()] = None


router = APIRouter(tags=["documents"])


@router.get("/documents/clients", response_model=None)
async def list_uploadable_clients(
    *,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    query: Annotated[DocumentQueryService, Depends(create_document_query_service)],
) -> list[ClientInfo]:
    """List clients available for client_private upload (assigned clients for internal, self for client)."""
    return await query.list_uploadable_clients(current_user.id, current_user.kind, current_user.role)


@router.post(
    "/documents",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.UPLOAD))],
)
async def upload_document(
    *,
    request: Request,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.DOCUMENTS_MANAGE))],
    options: Annotated[DocumentUploadForm, Depends()],
    cmd: Annotated[DocumentCommandService, Depends(create_document_command_service)],
    submission: Annotated[JobSubmission, Depends(create_job_submission)],
    upload_cfg: Annotated[UploadConfig, Depends(create_upload_config)],
    idempotency: Annotated[IdempotencyContext, Depends(create_idempotency_context)],
) -> UploadStatusResponse:
    filename = options.file.filename or "unnamed"

    if options.doc_domain is not None and options.doc_domain not in [d.value for d in DocDomain]:
        raise HTTPException(status_code=400, detail="doc_domain must be 'legal' or 'general'")

    max_bytes = upload_cfg.max_upload_size_mb * 1024 * 1024
    data = await read_upload_with_limit(options.file, max_bytes)

    await asyncio.to_thread(validate_uploaded_file, data, filename)

    cached, reservation = await reserve_idempotency(
        idempotency.store,
        idempotency.key,
        f"{current_user.kind}:{current_user.role}:{current_user.id}",
        f"{request.method}:{request.url.path}",
        {
            "filename": filename,
            "file_sha256": hashlib.sha256(data).hexdigest(),
            "visibility": options.visibility.value,
            "group_id": options.group_id,
            "client_id": options.client_id,
            "rename_on_conflict": options.rename_on_conflict,
            "doc_domain": options.doc_domain,
            "replaces_document_id": options.replaces_document_id,
        },
    )

    if cached is not None:
        return UploadStatusResponse(**cached)

    result = await upload_and_enqueue(
        file_data=data,
        filename=filename,
        visibility=options.visibility.value,
        group_id=options.group_id,
        client_id=options.client_id,
        user_id=current_user.id,
        user_kind=current_user.kind,
        user_role=current_user.role,
        rename_on_conflict=options.rename_on_conflict,
        doc_domain=options.doc_domain,
        replaces_document_id=options.replaces_document_id,
        document_service=cmd,
        action_name="document.upload",
        log_fn=submission.log,
    )

    response = UploadStatusResponse(
        status=result["status"], document_id=result["document_id"], filename=filename
    )

    # Idempotency: store result for future duplicate requests
    await complete_idempotency(idempotency.store, reservation, response.model_dump(mode="json"))

    return response


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    *,
    limit: Annotated[
        int, Query(ge=1, le=1000, description="Page size (M-6: unbounded lists are forbidden)")
    ] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.DOCUMENTS_VIEW))],
    query: Annotated[DocumentQueryService, Depends(create_document_query_service)],
) -> list[DocumentDTO]:
    return await query.list_documents(
        user_id=current_user.id,
        user_kind=current_user.kind,
        user_role=current_user.role,
        limit=limit,
        offset=offset,
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document_status(
    *,
    document_id: int,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.DOCUMENTS_VIEW))],
    query: Annotated[DocumentQueryService, Depends(create_document_query_service)],
) -> DocumentDTO:
    return await query.get_document(document_id, current_user.id, current_user.kind, current_user.role)


@router.delete(
    "/documents/{document_id}",
    status_code=202,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def delete_document(
    *,
    document_id: int,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.DOCUMENTS_MANAGE))],
    deletion_jobs: Annotated[DocumentDeletionJobs, Depends(create_document_deletion_jobs)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> DeleteDocumentResponse:
    job_id = await deletion_jobs.enqueue(document_id, current_user.id, current_user.kind, current_user.role)
    log(
        "document.delete.queued",
        user_id=current_user.id,
        details={"document_id": document_id, "job_id": job_id},
    )
    return DeleteDocumentResponse(status=BackgroundJobStatus.PENDING, document_id=document_id, job_id=job_id)


@router.get("/documents/{document_id}/deletion/{job_id}")
async def get_document_deletion(
    *,
    document_id: int,
    job_id: int,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.DOCUMENTS_MANAGE))],
    deletion_jobs: Annotated[DocumentDeletionJobs, Depends(create_document_deletion_jobs)],
) -> DeleteDocumentResponse:
    job = await deletion_jobs.get(document_id, job_id, current_user.id, current_user.kind, current_user.role)
    return DeleteDocumentResponse(status=job.status, document_id=document_id, job_id=job_id)


@router.patch(
    "/documents/{document_id}/rename",
    response_model=DocumentResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def rename_document(
    *,
    document_id: int,
    body: DocumentRenameRequest,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.DOCUMENTS_MANAGE))],
    cmd: Annotated[DocumentCommandService, Depends(create_document_command_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> DocumentDTO:
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
