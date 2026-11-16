"""Ingestion endpoints — thin wrappers around IngestionService (S3-only)."""

import asyncio
from dataclasses import dataclass
import hashlib
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from application.ingestion import IngestionPort
from application.jobs.service import JobService
from application.ports.idempotency_store import IdempotencyStorePort
from application.ports.rate_limit import RateLimitPolicyName
from application.ports.route_configs import UploadConfig
from application.spaces import KnowledgeSpaceService
from domain.knowledge.enums import DocumentVisibility, DomainDetectionMode
from domain.shared.exceptions import ValidationError
from presentation.api.auth_dependencies import require_admin
from presentation.api.constants import JobType
from presentation.api.dependencies import (
    IdempotencyContext,
    JobSubmission,
    create_idempotency_context,
    create_idempotency_store,
    create_ingest_service,
    create_ingestion_port,
    create_job_service,
    create_job_submission,
    create_knowledge_space_service,
    create_upload_config,
    get_idempotency_key,
)
from presentation.api.helpers import read_upload_with_limit
from presentation.api.idempotency import complete_idempotency, release_on_admission_error, reserve_idempotency
from presentation.api.rate_limit import rate_limit
from presentation.api.schemas import (
    CurrentUser,
    IngestRegistryItem,
    IngestRegistryResponse,
    IngestStatusResponse,
    SafeRelativePath,
    UploadResponse,
)
from presentation.api.schemas.validators import validate_uploaded_file


logger = logging.getLogger("default")


@dataclass(kw_only=True)
class IngestOptions:
    docs_dir: SafeRelativePath = "docs/"
    reset: bool = False
    # Profile key of the target space's pack or ``auto``.
    domain: str = DomainDetectionMode.AUTO.value
    visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC
    group_id: int | None = None
    client_id: int | None = None
    # Knowledge space of new documents; the only space when omitted.
    space_id: int | None = None


@dataclass(kw_only=True)
class IngestFileOptions:
    file_path: SafeRelativePath
    force: bool = False
    # Profile key of the target space's pack or ``auto``.
    domain: str = DomainDetectionMode.AUTO.value
    visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC
    group_id: int | None = None
    client_id: int | None = None
    # Knowledge space of new documents; the only space when omitted.
    space_id: int | None = None


router = APIRouter(tags=["ingest"])


async def validate_ingest_domain(spaces: KnowledgeSpaceService, space_id: int | None, domain: str) -> None:
    """Resolve the target space; accept auto-detection or a document profile of the space's pack."""
    profiles = await spaces.document_profiles(space_id)

    if domain != DomainDetectionMode.AUTO.value and domain not in {profile.key for profile in profiles}:
        known = ", ".join(profile.key for profile in profiles)
        message = f"Unknown document domain {domain!r}; known: {known}"
        raise ValidationError(message, errors={"domain": message})


MAX_UPLOAD_FILES = 20


@router.post(
    "/ingest",
    description="Reindex the selected prefix recursively. Existing IDs and access assignments are preserved; "
    "visibility, client and group options apply only to new documents.",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.UPLOAD))],
)
async def ingest_documents(
    *,
    request: Request,
    options: Annotated[IngestOptions, Depends()],
    admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[IngestionPort, Depends(create_ingest_service)],
    spaces: Annotated[KnowledgeSpaceService, Depends(create_knowledge_space_service)],
    submission: Annotated[JobSubmission, Depends(create_job_submission)],
    idempotency: Annotated[IdempotencyContext, Depends(create_idempotency_context)],
) -> IngestStatusResponse:
    await validate_ingest_domain(spaces, options.space_id, options.domain)

    try:
        resolved = service.resolve_docs_dir(options.docs_dir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    cached, reservation = await reserve_idempotency(
        idempotency.store,
        idempotency.key,
        f"{admin.kind}:{admin.role}:{admin.id}",
        f"{request.method}:{request.url.path}",
        {
            "docs_dir": resolved,
            "reset": options.reset,
            "domain": options.domain,
            "visibility": options.visibility.value,
            "group_id": options.group_id,
            "client_id": options.client_id,
            "space_id": options.space_id,
        },
    )

    if cached is not None:
        return IngestStatusResponse(**cached)

    async with release_on_admission_error(idempotency.store, reservation):
        job_id = await submission.jobs.create_job(
            JobType.INGEST,
            cost_principal=f"user:{admin.id}",
            dispatch_payload={
                "resolved_dir": resolved,
                "reset": options.reset,
                "domain": options.domain,
                "visibility": options.visibility.value,
                "group_id": options.group_id,
                "client_id": options.client_id,
                "space_id": options.space_id,
            },
        )

    submission.log(
        "ingest.full",
        user_id=admin.id,
        details={
            "docs_dir": resolved,
            "reset": options.reset,
            "domain": options.domain,
            "visibility": options.visibility.value,
        },
    )

    try:
        await submission.enqueuer.enqueue_ingest(
            resolved_dir=resolved,
            reset=options.reset,
            domain=options.domain,
            job_id=job_id,
            visibility=options.visibility.value,
            group_id=options.group_id,
            client_id=options.client_id,
            space_id=options.space_id,
        )
    except Exception:  # noqa: BLE001 — committed work remains available to the relay
        logger.warning("Ingestion delivery deferred: job_id=%d", job_id)
    mode = (
        "RESET selected prefix (preserve existing assignments)"
        if options.reset
        else "APPEND (new files only)"
    )
    response = IngestStatusResponse(status="started", mode=mode, docs_dir=resolved, job_id=job_id)
    await complete_idempotency(idempotency.store, reservation, response.model_dump(mode="json"))
    return response


@router.post(
    "/ingest/file",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.UPLOAD))],
)
async def ingest_single_file(
    *,
    request: Request,
    options: Annotated[IngestFileOptions, Depends()],
    admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[IngestionPort, Depends(create_ingest_service)],
    spaces: Annotated[KnowledgeSpaceService, Depends(create_knowledge_space_service)],
    submission: Annotated[JobSubmission, Depends(create_job_submission)],
    idempotency: Annotated[IdempotencyContext, Depends(create_idempotency_context)],
) -> IngestStatusResponse:
    await validate_ingest_domain(spaces, options.space_id, options.domain)

    try:
        resolved = service.resolve_ingest_target(options.file_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    cached, reservation = await reserve_idempotency(
        idempotency.store,
        idempotency.key,
        f"{admin.kind}:{admin.role}:{admin.id}",
        f"{request.method}:{request.url.path}",
        {
            "file_path": resolved,
            "force": options.force,
            "domain": options.domain,
            "visibility": options.visibility.value,
            "group_id": options.group_id,
            "client_id": options.client_id,
            "space_id": options.space_id,
        },
    )

    if cached is not None:
        return IngestStatusResponse(**cached)

    async with release_on_admission_error(idempotency.store, reservation):
        job_id = await submission.jobs.create_job(
            JobType.INGEST_FILE,
            cost_principal=f"user:{admin.id}",
            dispatch_payload={
                "resolved": resolved,
                "force": options.force,
                "domain": options.domain,
                "visibility": options.visibility.value,
                "group_id": options.group_id,
                "client_id": options.client_id,
                "space_id": options.space_id,
            },
        )

    submission.log(
        "ingest.file",
        user_id=admin.id,
        details={
            "file": resolved,
            "force": options.force,
            "domain": options.domain,
            "visibility": options.visibility.value,
        },
    )

    try:
        await submission.enqueuer.enqueue_ingest_file(
            resolved=resolved,
            domain=options.domain,
            job_id=job_id,
            visibility=options.visibility.value,
            group_id=options.group_id,
            client_id=options.client_id,
            space_id=options.space_id,
            force=options.force,
        )
    except Exception:  # noqa: BLE001 — committed work remains available to the relay
        logger.warning("Ingestion delivery deferred: job_id=%d", job_id)
    response = IngestStatusResponse(status="started", file=resolved, force=options.force, job_id=job_id)
    await complete_idempotency(idempotency.store, reservation, response.model_dump(mode="json"))
    return response


@router.get("/ingest/registry")
async def get_ingest_registry(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[IngestionPort, Depends(create_ingest_service)],
) -> IngestRegistryResponse:
    result = await service.get_registry()
    return IngestRegistryResponse(
        total_files=result.total_files,
        total_chunks=result.total_chunks,
        files=[
            IngestRegistryItem(
                filename=i.filename,
                ingestion_key=i.ingestion_key,
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
    dependencies=[Depends(rate_limit(RateLimitPolicyName.UPLOAD))],
)
async def upload_files(  # noqa: PLR0913 -- explicit HTTP dependencies
    *,
    request: Request,
    files: Annotated[list[UploadFile], File()],
    admin: Annotated[CurrentUser, Depends(require_admin)],
    ingestion_port: Annotated[IngestionPort, Depends(create_ingestion_port)],
    upload_cfg: Annotated[UploadConfig, Depends(create_upload_config)],
    idempotency_key: Annotated[str | None, Depends(get_idempotency_key)],
    idempotency_store: Annotated[IdempotencyStorePort, Depends(create_idempotency_store)],
    jobs: Annotated[JobService, Depends(create_job_service)],
) -> UploadResponse:
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
        fc = await asyncio.to_thread(validate_uploaded_file, data, f.filename or "unnamed")
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

    async with release_on_admission_error(idempotency_store, reservation):
        await jobs.check_capacity(f"user:{admin.id}")

    uploaded = await ingestion_port.upload_files(file_data)
    response = UploadResponse(files=uploaded)
    await complete_idempotency(idempotency_store, reservation, response.model_dump(mode="json"))
    return response
