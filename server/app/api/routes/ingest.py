"""
api/routes/ingest.py — Ingestion endpoints. Thin wrappers around IngestService.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from infrastructure.auth import require_admin
from services.ingest_service import IngestService

from api.schemas import IngestRegistryItem, IngestRegistryResponse, IngestStatusResponse, UploadResponse

logger = logging.getLogger("default")

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestStatusResponse)
async def ingest_documents(
    background_tasks: BackgroundTasks,
    docs_dir: str = "/code/project/data/docs_sample",
    reset: bool = False,
    admin: dict = Depends(require_admin),
):
    service = IngestService()
    try:
        resolved_dir = service.resolve_docs_dir(docs_dir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    background_tasks.add_task(service.run_full_ingestion, resolved_dir, reset)
    mode = "RESET + full reindex" if reset else "APPEND (new files only)"
    return IngestStatusResponse(status="started", mode=mode, docs_dir=resolved_dir)


@router.post("/ingest/file", response_model=IngestStatusResponse)
async def ingest_single_file(
    background_tasks: BackgroundTasks,
    file_path: str,
    force: bool = False,
    admin: dict = Depends(require_admin),
):
    service = IngestService()
    try:
        resolved = service.resolve_ingest_target(file_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if force:
        service.force_reindex(Path(resolved).name)
    background_tasks.add_task(service.run_single_file, resolved)
    return IngestStatusResponse(status="started", file=resolved, force=force)


@router.get("/ingest/registry", response_model=IngestRegistryResponse)
async def get_ingest_registry(admin: dict = Depends(require_admin)):
    service = IngestService()
    registry = service.get_registry()
    items = [
        IngestRegistryItem(
            filename=name,
            chunks=meta.get("chunks", 0),
            chars=meta.get("chars", 0),
            indexed_at=meta.get("indexed_at", ""),
            source=meta.get("source", ""),
        )
        for name, meta in sorted(registry.items())
    ]
    return IngestRegistryResponse(
        total_files=len(items),
        total_chunks=sum(i.chunks for i in items),
        files=items,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: list[UploadFile] = File(...),
    admin: dict = Depends(require_admin),
):
    service = IngestService()

    file_data = []
    for f in files:
        data = await f.read()
        file_data.append(type("UploadFileData", (), {"filename": f.filename, "data": data})())

    uploaded = service.upload_files(file_data)
    return UploadResponse(files=uploaded)
