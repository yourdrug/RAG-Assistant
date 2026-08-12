"""Document endpoints — thin wrappers around DocumentService."""

from __future__ import annotations

import logging
from pathlib import Path

from application.services.document_processor import DocumentProcessor
from application.services.document_service import DocumentService
from config import settings
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from infrastructure.logging.actions import log_action

from presentation.api.auth_dependencies import get_current_user
from presentation.api.dependencies import (
    create_document_service,
    get_document_parser,
    get_document_splitter,
    get_file_storage,
    get_uow_factory,
    get_vector_store_repo,
)
from presentation.api.routes.common import create_background_job
from presentation.api.schemas import DocumentResponse, UploadStatusResponse

logger = logging.getLogger("default")

router = APIRouter(tags=["documents"])

# Magic-byte signatures for supported extensions
_MAGIC_BYTES: dict[str, list[bytes]] = {
    ".pdf": [b"%PDF"],
    ".docx": [b"PK\x03\x04"],  # ZIP-based
    ".doc": [b"\xd0\xcf\x11\xe0"],  # OLE2
    ".rtf": [b"{\\rtf"],
    ".md": [],  # plain text, no reliable magic
    ".txt": [],  # plain text
}


def _validate_mime(file_data: bytes, extension: str) -> None:
    """Verify file contents match declared extension using magic bytes."""
    if extension not in _MAGIC_BYTES:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {extension}")
    expected = _MAGIC_BYTES[extension]
    if not expected:
        return  # plain text — no magic to check
    if not any(file_data[: len(sig)] == sig for sig in expected):
        raise HTTPException(
            status_code=400,
            detail=f"File content does not match extension {extension}",
        )


async def _process_document_in_background(
    document_id: int,
    storage_key: str,
    filename: str,
    visibility: str,
    owner_id: int | None,
    group_id: int | None,
    replace_id: int | None,
    job_id: int,
    doc_domain: str | None = None,
):
    """Async — runs on the event loop after the response is sent."""
    uow_factory = get_uow_factory()

    try:
        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_running(job_id)

        processor = DocumentProcessor(
            uow_factory=uow_factory,
            vector_store_repo=get_vector_store_repo(),
            file_storage=get_file_storage(),
            document_parser=get_document_parser(),
            document_splitter=get_document_splitter(),
        )

        logger.info("Background upload started: %s (doc %d, job %d)", filename, document_id, job_id)
        await processor.process(
            document_id=document_id,
            storage_key=storage_key,
            original_filename=filename,
            visibility=visibility,
            owner_id=owner_id,
            group_id=group_id,
            replace_id=replace_id,
            doc_domain=doc_domain,
        )
        logger.info("Background upload completed: %s (doc %d, job %d)", filename, document_id, job_id)
        async with uow_factory.create(master=True) as uow:
            await uow.background_jobs.mark_done(job_id)
    except Exception as e:
        logger.exception(
            "Background document processing failed for %s (doc %d, job %d)", filename, document_id, job_id
        )
        try:
            async with uow_factory.create(master=True) as uow:
                await uow.background_jobs.mark_failed(job_id, str(e)[:500])
        except Exception:
            logger.exception("Failed to mark job %d as failed", job_id)


@router.get("/documents/clients")
async def list_uploadable_clients(
    current_user: dict = Depends(get_current_user),
    document_service: DocumentService = Depends(create_document_service),
):
    """List clients available for client_private upload (assigned clients for internal, self for client)."""
    return await document_service.list_uploadable_clients(
        current_user["id"], current_user["kind"], current_user["role"]
    )


@router.post("/documents", response_model=UploadStatusResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    file: UploadFile = File(...),
    visibility: str = Form(...),
    group_id: int | None = Form(None),
    client_id: int | None = Form(None),
    rename_on_conflict: bool = Form(False),
    doc_domain: str | None = Form(None),
    document_service: DocumentService = Depends(create_document_service),
):
    filename = file.filename or "unnamed"
    ext = Path(filename).suffix.lower()

    if doc_domain is not None and doc_domain not in ("legal", "general"):
        raise HTTPException(status_code=400, detail="doc_domain must be 'legal' or 'general'")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(data) / 1024 / 1024:.1f} MB (limit {settings.max_upload_size_mb} MB)",
        )

    _validate_mime(data, ext)

    result = await document_service.upload(
        filename=filename,
        file_data=data,
        visibility=visibility,
        group_id=group_id,
        client_id=client_id,
        user_id=current_user["id"],
        user_kind=current_user["kind"],
        user_role=current_user["role"],
        rename_on_conflict=rename_on_conflict,
        doc_domain=doc_domain,
    )

    log_action(
        "document.upload",
        user_id=current_user["id"],
        details={"filename": filename, "visibility": visibility},
    )

    job_id = await create_background_job(get_uow_factory(), "document_processing", related_id=result.id)

    background_tasks.add_task(
        _process_document_in_background,
        document_id=result.id,
        storage_key=result.storage_key,
        filename=result.filename,
        visibility=visibility,
        owner_id=result.owner_id,
        group_id=group_id,
        replace_id=result.replace_id,
        doc_domain=doc_domain,
        job_id=job_id,
    )

    return UploadStatusResponse(status="processing", document_id=result.id, filename=filename)


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
