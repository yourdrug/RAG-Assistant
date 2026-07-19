import logging
from pathlib import Path

from config import settings
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from infrastructure.auth import require_admin
from infrastructure.vector_store import (
    load_registry,
    run_ingest_file,
    run_ingestion,
    save_registry,
)

from api.schemas import UploadResponse

logger = logging.getLogger("default")

router = APIRouter(tags=["ingest"])


@router.post("/ingest")
async def ingest_documents(
    background_tasks: BackgroundTasks,
    docs_dir: str = f"{settings.data_dir}/docs_sample",
    reset: bool = False,
    admin: dict = Depends(require_admin),
):
    background_tasks.add_task(run_ingestion, docs_dir, reset)
    mode = "RESET + full reindex" if reset else "APPEND (new files only)"
    return {"status": "started", "mode": mode, "docs_dir": docs_dir}


def _resolve_ingest_target(file_path: str) -> str:
    if settings.file_backend == "s3":
        if file_path.startswith("/") or ".." in Path(file_path).parts:
            raise HTTPException(400, "file_path не должен содержать '..' или быть абсолютным (S3 key)")
        return file_path

    base = Path(settings.data_dir).resolve()
    candidate = Path(file_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"file_path должен быть внутри {base} (DATA_DIR)"
        ) from None
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return str(resolved)


@router.post("/ingest/file")
async def ingest_single_file(
    background_tasks: BackgroundTasks,
    file_path: str,
    force: bool = False,
    admin: dict = Depends(require_admin),
):
    resolved = _resolve_ingest_target(file_path)

    if force:
        reg = load_registry()
        reg.pop(Path(resolved).name, None)
        save_registry(reg)
    background_tasks.add_task(run_ingest_file, resolved)
    return {"status": "started", "file": resolved, "force": force}


@router.get("/ingest/registry")
async def get_ingest_registry(admin: dict = Depends(require_admin)):
    registry = load_registry()
    items = [
        {
            "filename": name,
            "chunks": meta.get("chunks", 0),
            "chars": meta.get("chars", 0),
            "indexed_at": meta.get("indexed_at", ""),
            "source": meta.get("source", ""),
        }
        for name, meta in sorted(registry.items())
    ]
    return {
        "total_files": len(items),
        "total_chunks": sum(i["chunks"] for i in items),
        "files": items,
    }


@router.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: list[UploadFile] = File(...),
    admin: dict = Depends(require_admin),
):
    from infrastructure.storage import get_storage

    storage = get_storage()
    prefix = "docs/"
    uploaded: list[str] = []
    for f in files:
        key = prefix + f.filename
        data = await f.read()
        storage.upload_file(key, data)
        uploaded.append(key)
        logger.info("Uploaded: %s (%d bytes)", key, len(data))
    return UploadResponse(files=uploaded)
