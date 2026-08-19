"""Ingestion endpoints — thin wrappers around IngestAppService."""

from __future__ import annotations

import logging
from pathlib import Path

from application.ports.ingestion_port import IngestionPort
from application.services.ingest_service import IngestAppService
from application.services.job_service import JobService
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from infrastructure.logging.actions import log_action
from infrastructure.worker.queue import enqueue_ingest, enqueue_ingest_file

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import create_ingest_service, create_ingestion_port, create_job_service
from presentation.api.rate_limits import ingest_rate_limit
from presentation.api.schemas import (
    IngestRegistryItem,
    IngestRegistryResponse,
    IngestStatusResponse,
    UploadResponse,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestStatusResponse, dependencies=[Depends(ingest_rate_limit)])
async def ingest_documents(
    docs_dir: str = "/code/project/data/docs_sample",
    reset: bool = False,
    domain: str = "auto",
    admin: dict = Depends(require_admin),
    service: IngestAppService = Depends(create_ingest_service),
    job_service: JobService = Depends(create_job_service),
):
    try:
        resolved_dir = service.resolve_docs_dir(docs_dir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    job_id = await job_service.create_job("ingest")

    log_action(
        "ingest.full",
        user_id=admin["id"],
        details={"docs_dir": resolved_dir, "reset": reset, "domain": domain},
    )

    await enqueue_ingest(
        resolved_dir=resolved_dir,
        reset=reset,
        domain=domain,
        job_id=job_id,
    )
    mode = "RESET + full reindex" if reset else "APPEND (new files only)"
    return IngestStatusResponse(status="started", mode=mode, docs_dir=resolved_dir)


@router.post("/ingest/file", response_model=IngestStatusResponse, dependencies=[Depends(ingest_rate_limit)])
async def ingest_single_file(
    file_path: str,
    force: bool = False,
    domain: str = "auto",
    admin: dict = Depends(require_admin),
    service: IngestAppService = Depends(create_ingest_service),
    job_service: JobService = Depends(create_job_service),
):
    try:
        resolved = service.resolve_ingest_target(file_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    if force:
        service.force_reindex(Path(resolved).name)

    job_id = await job_service.create_job("ingest", related_id=None)

    log_action(
        "ingest.file", user_id=admin["id"], details={"file": resolved, "force": force, "domain": domain}
    )

    await enqueue_ingest_file(
        resolved=resolved,
        domain=domain,
        job_id=job_id,
    )
    return IngestStatusResponse(status="started", file=resolved, force=force)


@router.get("/ingest/registry", response_model=IngestRegistryResponse)
async def get_ingest_registry(
    admin: dict = Depends(require_admin),
    service: IngestAppService = Depends(create_ingest_service),
):
    result = service.get_registry()
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


@router.post("/upload", response_model=UploadResponse, dependencies=[Depends(ingest_rate_limit)])
async def upload_files(
    files: list[UploadFile] = File(...),
    admin: dict = Depends(require_admin),
    ingestion_port: IngestionPort = Depends(create_ingestion_port),
):
    file_data = []
    for f in files:
        data = await f.read()
        file_data.append(type("UploadFileData", (), {"filename": f.filename, "data": data})())

    uploaded = ingestion_port.upload_files(file_data)
    return UploadResponse(files=uploaded)
