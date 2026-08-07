"""Ingestion endpoints — thin wrappers around IngestAppService."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from infrastructure.logging.actions import log_action

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import create_ingest_service, create_ingestion_service, get_uow_factory
from presentation.api.routes.common import create_background_job
from presentation.api.schemas import (
    IngestRegistryItem,
    IngestRegistryResponse,
    IngestStatusResponse,
    UploadResponse,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestStatusResponse)
async def ingest_documents(
    background_tasks: BackgroundTasks,
    docs_dir: str = "/code/project/data/docs_sample",
    reset: bool = False,
    admin: dict = Depends(require_admin),
):
    service = create_ingest_service()
    try:
        resolved_dir = service.resolve_docs_dir(docs_dir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job_id = await create_background_job(get_uow_factory(), "ingest")

    log_action("ingest.full", user_id=admin["id"], details={"docs_dir": resolved_dir, "reset": reset})

    async def _tracked_ingest():
        uow_factory = get_uow_factory()
        try:
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_running(job_id)
            await service.run_full(resolved_dir, reset)
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_done(job_id)
        except Exception as e:
            logger.exception("Background ingest failed (job %d)", job_id)
            try:
                async with uow_factory.create(master=True) as uow:
                    await uow.background_jobs.mark_failed(job_id, str(e)[:500])
            except Exception:
                logger.exception("Failed to mark job %d as failed", job_id)

    background_tasks.add_task(_tracked_ingest)
    mode = "RESET + full reindex" if reset else "APPEND (new files only)"
    return IngestStatusResponse(status="started", mode=mode, docs_dir=resolved_dir)


@router.post("/ingest/file", response_model=IngestStatusResponse)
async def ingest_single_file(
    background_tasks: BackgroundTasks,
    file_path: str,
    force: bool = False,
    admin: dict = Depends(require_admin),
):
    service = create_ingest_service()
    try:
        resolved = service.resolve_ingest_target(file_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if force:
        service.force_reindex(Path(resolved).name)

    job_id = await create_background_job(get_uow_factory(), "ingest", related_id=None)

    log_action("ingest.file", user_id=admin["id"], details={"file": resolved, "force": force})

    async def _tracked_single():
        uow_factory = get_uow_factory()
        try:
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_running(job_id)
            await service.run_single(resolved)
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_done(job_id)
        except Exception as e:
            logger.exception("Background single-file ingest failed (job %d)", job_id)
            try:
                async with uow_factory.create(master=True) as uow:
                    await uow.background_jobs.mark_failed(job_id, str(e)[:500])
            except Exception:
                logger.exception("Failed to mark job %d as failed", job_id)

    background_tasks.add_task(_tracked_single)
    return IngestStatusResponse(status="started", file=resolved, force=force)


@router.get("/ingest/registry", response_model=IngestRegistryResponse)
async def get_ingest_registry(admin: dict = Depends(require_admin)):
    service = create_ingest_service()
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


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: list[UploadFile] = File(...),
    admin: dict = Depends(require_admin),
):
    service = create_ingestion_service()

    file_data = []
    for f in files:
        data = await f.read()
        file_data.append(type("UploadFileData", (), {"filename": f.filename, "data": data})())

    uploaded = service.upload_files(file_data)
    return UploadResponse(files=uploaded)
