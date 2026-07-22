"""Document endpoints — thin wrappers around DocumentService."""

from __future__ import annotations

import logging

from application.services.document_processor import DocumentProcessor
from application.services.document_service import DocumentService
from application.uow import UnitOfWork
from application.use_cases.document.delete_document import DeleteDocument
from application.use_cases.document.get_document import GetDocument
from application.use_cases.document.list_documents import ListDocuments
from application.use_cases.document.upload_document import UploadDocument
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from infrastructure.auth.fastapi_dependencies import get_current_user
from infrastructure.repositories.qdrant_vector_store_repository import QdrantVectorStoreRepository
from infrastructure.storage import get_storage

from presentation.api.dependencies import get_uow
from presentation.api.schemas import DocumentResponse, UploadStatusResponse

logger = logging.getLogger("default")

router = APIRouter(tags=["documents"])


def _document_service(uow: UnitOfWork) -> DocumentService:
    return DocumentService(
        upload_document=UploadDocument(
            document_repo=uow.documents,
            group_repo=uow.groups,
            document_processor=DocumentProcessor(uow._session, uow.documents),
            file_storage=get_storage(),
        ),
        list_documents=ListDocuments(
            document_repo=uow.documents,
            group_repo=uow.groups,
            client_assignment_repo=uow.client_assignments,
        ),
        get_document=GetDocument(
            document_repo=uow.documents,
            group_repo=uow.groups,
            client_assignment_repo=uow.client_assignments,
        ),
        delete_document=DeleteDocument(
            document_repo=uow.documents,
            vector_store_repo=QdrantVectorStoreRepository(),
            file_storage=get_storage(),
            group_repo=uow.groups,
        ),
    )


def _process_upload_background(
    document_id: int,
    filename: str,
    visibility: str,
    owner_id: int | None,
    group_id: int | None,
    replace_id: int | None,
):
    from infrastructure.database.engine import SessionLocal
    from infrastructure.repositories.sqlalchemy_document_repository import SQLAlchemyDocumentRepository

    db = SessionLocal()
    try:
        doc_repo = SQLAlchemyDocumentRepository(db)
        processor = DocumentProcessor(db, doc_repo)

        storage_key = f"uploads/public/{document_id}_{filename}"
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
    except Exception as e:
        logger.exception("Background upload failed: %s (doc %d)", filename, document_id)
        try:
            doc_repo = SQLAlchemyDocumentRepository(db)
            doc_repo.update_status(document_id, "failed", error=str(e))
        except Exception:
            logger.exception("Failed to mark document as failed: %s", filename)
    finally:
        try:
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()


@router.post("/documents", response_model=UploadStatusResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    uow: UnitOfWork = Depends(get_uow),
    file: UploadFile = File(...),
    visibility: str = Form(...),
    group_id: int | None = Form(None),
):
    data = await file.read()
    filename = file.filename or "unnamed"

    service = _document_service(uow)
    result = await service.upload(
        filename=filename,
        file_data=data,
        visibility=visibility,
        group_id=group_id,
        user_id=current_user["id"],
        user_kind=current_user["kind"],
        user_role=current_user["role"],
    )

    return UploadStatusResponse(status="processing", document_id=result.id, filename=filename)


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    current_user: dict = Depends(get_current_user),
    uow: UnitOfWork = Depends(get_uow),
):
    service = _document_service(uow)
    return service.list_documents(current_user["id"], current_user["kind"])


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document_status(
    document_id: int,
    current_user: dict = Depends(get_current_user),
    uow: UnitOfWork = Depends(get_uow),
):
    service = _document_service(uow)
    return service.get_document(document_id, current_user["id"], current_user["kind"], current_user["role"])


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: dict = Depends(get_current_user),
    uow: UnitOfWork = Depends(get_uow),
):
    service = _document_service(uow)
    service.delete_document(document_id, current_user["id"], current_user["role"])
    return {"status": "deleted", "document_id": document_id}
