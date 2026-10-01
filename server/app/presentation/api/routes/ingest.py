"""Ingestion endpoints — thin wrappers around IngestAppService (S3-only)."""

from __future__ import annotations

import logging
import hashlib

from application.ports.ingestion_port import IngestionPort
from application.ports.rate_limit import RateLimitPolicyName
from application.services.ingest_service import IngestAppService
from application.services.job_service import JobService
from domain.value_objects.visibility import DocumentVisibility
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from presentation.api.auth_dependencies import require_admin
from presentation.api.rate_limit import rate_limit
from presentation.api.constants import JobType
from presentation.api.dependencies import (
    create_action_logger,
    create_ingest_service,
    create_ingestion_port,
    create_idempotency_store,
    create_job_enqueuer,
    create_job_service,
    create_upload_config,
    get_idempotency_key,
)
from presentation.api.helpers import read_upload_with_limit
from presentation.api.idempotency import complete_idempotency, reserve_idempotency
from presentation.api.schemas.validators import validate_uploaded_file
from presentation.api.schemas import (
    CurrentUser,
    IngestRegistryItem,
    IngestRegistryResponse,
    IngestStatusResponse,
    SafeRelativePath,
    UploadResponse,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["ingest"])

MAX_UPLOAD_FILES = 20


@router.post(
    "/ingest",
    response_model=IngestStatusResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.UPLOAD))],
)
async def ingest_documents(
    request: Request,
    docs_dir: SafeRelativePath = "docs/",
    reset: bool = False,
    domain: str = "auto",
    visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
    group_id: int | None = None,
    client_id: int | None = None,
    admin: CurrentUser = Depends(require_admin),
    service: IngestAppService = Depends(create_ingest_service),
    job_service: JobService = Depends(create_job_service),
    log=Depends(create_action_logger),
    job_enqueuer=Depends(create_job_enqueuer),
    idempotency_key: str | None = Depends(get_idempotency_key),
    idempotency_store=Depends(create_idempotency_store),
):
    try:
        resolved = service.resolve_docs_dir(docs_dir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    cached, reservation = await reserve_idempotency(
        idempotency_store,
        idempotency_key,
        f"{admin.kind}:{admin.role}:{admin.id}",
        f"{request.method}:{request.url.path}",
        {
            "docs_dir": resolved,
            "reset": reset,
            "domain": domain,
            "visibility": visibility.value,
            "group_id": group_id,
            "client_id": client_id,
        },
    )
    if cached is not None:
        return IngestStatusResponse(**cached)

    job_id = await job_service.create_job(JobType.INGEST)

    log(
        "ingest.full",
        user_id=admin.id,
        details={"docs_dir": resolved, "reset": reset, "domain": domain, "visibility": visibility.value},
    )

    await job_enqueuer.enqueue_ingest(
        resolved_dir=resolved,
        reset=reset,
        domain=domain,
        job_id=job_id,
        visibility=visibility.value,
        group_id=group_id,
        client_id=client_id,
    )
    mode = "RESET + full reindex" if reset else "APPEND (new files only)"
    response = IngestStatusResponse(status="started", mode=mode, docs_dir=resolved)
    await complete_idempotency(idempotency_store, reservation, response.model_dump(mode="json"))
    return response


@router.post(
    "/ingest/file",
    response_model=IngestStatusResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.UPLOAD))],
)
async def ingest_single_file(
    request: Request,
    file_path: SafeRelativePath,
    force: bool = False,
    domain: str = "auto",
    visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
    group_id: int | None = None,
    client_id: int | None = None,
    admin: CurrentUser = Depends(require_admin),
    service: IngestAppService = Depends(create_ingest_service),
    job_service: JobService = Depends(create_job_service),
    log=Depends(create_action_logger),
    job_enqueuer=Depends(create_job_enqueuer),
    idempotency_key: str | None = Depends(get_idempotency_key),
    idempotency_store=Depends(create_idempotency_store),
):
    try:
        resolved = service.resolve_ingest_target(file_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    cached, reservation = await reserve_idempotency(
        idempotency_store,
        idempotency_key,
        f"{admin.kind}:{admin.role}:{admin.id}",
        f"{request.method}:{request.url.path}",
        {
            "file_path": resolved,
            "force": force,
            "domain": domain,
            "visibility": visibility.value,
            "group_id": group_id,
            "client_id": client_id,
        },
    )
    if cached is not None:
        return IngestStatusResponse(**cached)

    if force:
        await service.force_reindex(file_path.split("/")[-1])

    job_id = await job_service.create_job(JobType.INGEST, related_id=None)

    log(
        "ingest.file",
        user_id=admin.id,
        details={"file": resolved, "force": force, "domain": domain, "visibility": visibility.value},
    )

    await job_enqueuer.enqueue_ingest_file(
        resolved=resolved,
        domain=domain,
        job_id=job_id,
        visibility=visibility.value,
        group_id=group_id,
        client_id=client_id,
    )
    response = IngestStatusResponse(status="started", file=resolved, force=force)
    await complete_idempotency(idempotency_store, reservation, response.model_dump(mode="json"))
    return response


@router.get("/ingest/registry", response_model=IngestRegistryResponse)
async def get_ingest_registry(
    admin: CurrentUser = Depends(require_admin),
    service: IngestAppService = Depends(create_ingest_service),
):
    result = await service.get_registry()
    return IngestRegistryResponse(
        total_files=result.total_files,
        total_chunks=result.total_chunks,
        files=[
            IngestRegistryItem(
                filename=i.filename,
                chunks=i.chunks,
                chars=i.chars,
                indexed_at=i.indexed_at,
                source=i.source,
            )
            for i in result.files
        ],
    )


@router.post(
    "/upload",
    response_model=UploadResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.UPLOAD))],
)
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    admin: CurrentUser = Depends(require_admin),
    ingestion_port: IngestionPort = Depends(create_ingestion_port),
    upload_cfg=Depends(create_upload_config),
    idempotency_key: str | None = Depends(get_idempotency_key),
    idempotency_store=Depends(create_idempotency_store),
):
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files: {len(files)} (limit {MAX_UPLOAD_FILES})",
        )

    max_bytes = upload_cfg.max_upload_size_mb * 1024 * 1024
    file_data = []
    file_fingerprints = []
    for f in files:
        data = await read_upload_with_limit(f, max_bytes)
        fc = validate_uploaded_file(data, f.filename or "unnamed")
        file_data.append(type("UploadFileData", (), {"filename": f.filename, "data": fc.data})())
        file_fingerprints.append(
            {"filename": f.filename or "unnamed", "sha256": hashlib.sha256(fc.data).hexdigest()}
        )

    cached, reservation = await reserve_idempotency(
        idempotency_store,
        idempotency_key,
        f"{admin.kind}:{admin.role}:{admin.id}",
        f"{request.method}:{request.url.path}",
        {"files": file_fingerprints},
    )
    if cached is not None:
        return UploadResponse(**cached)

    uploaded = await ingestion_port.upload_files(file_data)
    response = UploadResponse(files=uploaded)
    await complete_idempotency(idempotency_store, reservation, response.model_dump(mode="json"))
    return response
