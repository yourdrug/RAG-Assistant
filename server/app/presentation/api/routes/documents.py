"""Document endpoints — thin wrappers around DocumentService."""

from __future__ import annotations

import logging

from application.services.document_processor import DocumentProcessor
from application.services.document_service import DocumentService
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from infrastructure.auth.fastapi_dependencies import get_current_user
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser, LangchainDocumentSplitter
from infrastructure.repositories.qdrant_vector_store_repository import QdrantVectorStoreRepository
from infrastructure.storage import get_storage
from infrastructure.uow_factory import UnitOfWorkFactory

from presentation.api.schemas import DocumentResponse, UploadStatusResponse

logger = logging.getLogger("default")

router = APIRouter(tags=["documents"])

_uow_factory = UnitOfWorkFactory()
_vector_store_repo = QdrantVectorStoreRepository()
_file_storage = get_storage()
_document_parser = LangchainDocumentParser()
_document_splitter = LangchainDocumentSplitter()

_document_service = DocumentService(
    uow_factory=_uow_factory,
    vector_store_repo=_vector_store_repo,
    file_storage=_file_storage,
)


def _process_document_in_background(
    document_id: int,
    storage_key: str,
    filename: str,
    visibility: str,
    owner_id: int | None,
    group_id: int | None,
    replace_id: int | None,
):
    """Run after the HTTP response is sent and the document record is committed.

    DocumentProcessor manages its own UoW internally — one transaction for the
    entire processing pipeline (parse → embed → upload to vector store → update status).
    """
    processor = DocumentProcessor(
        uow_factory=_uow_factory,
        vector_store_repo=_vector_store_repo,
        file_storage=_file_storage,
        document_parser=_document_parser,
        document_splitter=_document_splitter,
    )

    logger.info("Background upload started: %s (doc %d)", filename, document_id)
    processor.process(
        document_id=document_id,
        storage_key=storage_key,
        original_filename=filename,
        visibility=visibility,
        owner_id=owner_id,
        group_id=group_id,
        replace_id=replace_id,
    )
    logger.info("Background upload completed: %s (doc %d)", filename, document_id)


@router.post("/documents", response_model=UploadStatusResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    file: UploadFile = File(...),
    visibility: str = Form(...),
    group_id: int | None = Form(None),
    rename_on_conflict: bool = Form(False),
):
    """Upload a document.

    The UoW commits the document record to DB BEFORE the background task starts,
    so the processor always sees a committed document in a fresh session.

    If rename_on_conflict=true and a file with the same name exists, the new file
    is auto-renamed (e.g. readme.md -> readme(1).md) instead of replacing it.
    """
    data = await file.read()
    filename = file.filename or "unnamed"

    result = await _document_service.upload(
        filename=filename,
        file_data=data,
        visibility=visibility,
        group_id=group_id,
        user_id=current_user["id"],
        user_kind=current_user["kind"],
        user_role=current_user["role"],
        rename_on_conflict=rename_on_conflict,
    )
    background_tasks.add_task(
        _process_document_in_background,
        document_id=result.id,
        storage_key=result.storage_key,
        filename=result.filename,
        visibility=visibility,
        owner_id=result.owner_id,
        group_id=group_id,
        replace_id=result.replace_id,
    )

    return UploadStatusResponse(status="processing", document_id=result.id, filename=filename)


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    current_user: dict = Depends(get_current_user),
):
    return _document_service.list_documents(current_user["id"], current_user["kind"])


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document_status(
    document_id: int,
    current_user: dict = Depends(get_current_user),
):
    return _document_service.get_document(
        document_id, current_user["id"], current_user["kind"], current_user["role"]
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: dict = Depends(get_current_user),
):
    _document_service.delete_document(document_id, current_user["id"], current_user["role"])
    return {"status": "deleted", "document_id": document_id}
