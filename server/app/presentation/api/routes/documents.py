"""Document endpoints — thin wrappers around DocumentService."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from infrastructure.auth.fastapi_dependencies import get_current_user
from infrastructure.database.engine import SessionLocal
from infrastructure.database.session import get_db
from sqlalchemy.orm import Session

from presentation.api.schemas import DocumentResponse, UploadStatusResponse

logger = logging.getLogger("default")

router = APIRouter(tags=["documents"])


def _process_upload_background(
    document_id: int,
    filename: str,
    visibility: str,
    owner_id: int | None,
    group_id: int | None,
    replace_id: int | None,
):
    db = SessionLocal()
    try:
        from infrastructure.repositories.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
        from infrastructure.services.document_processor import DocumentProcessor

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
        db.commit()
        logger.info("Background upload completed: %s (doc %d)", filename, document_id)
    except Exception as e:
        logger.exception("Background upload failed: %s (doc %d)", filename, document_id)
        try:
            db.rollback()
        except Exception:
            pass
        try:
            from infrastructure.repositories.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
            doc_repo = SQLAlchemyDocumentRepository(db)
            doc_repo.update_status(document_id, "failed", error=str(e))
            db.commit()
        except Exception:
            logger.exception("Failed to mark document as failed: %s", filename)
    finally:
        db.close()


@router.post("/documents", response_model=UploadStatusResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    visibility: str = Form(...),
    group_id: int | None = Form(None),
):
    from domain.entities.document import Document
    from domain.services.access_control import compute_owner_and_group, validate_document_visibility
    from domain.value_objects.roles import UserKind, UserRole
    from domain.value_objects.visibility import DocumentVisibility
    from infrastructure.repositories.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
    from infrastructure.repositories.sqlalchemy_group_repository import SQLAlchemyGroupRepository
    from infrastructure.storage import get_storage

    data = await file.read()
    filename = file.filename or "unnamed"

    vis = DocumentVisibility.validate(visibility)
    user_kind = UserKind(current_user["kind"])
    user_role = UserRole(current_user["role"])

    group_repo = SQLAlchemyGroupRepository(db)
    user_group_ids = group_repo.get_user_group_ids(current_user["id"])

    validate_document_visibility(vis, group_id, user_kind, user_role, user_group_ids)

    if vis == DocumentVisibility.INTERNAL_GROUP:
        groups = group_repo.list_by_ids([group_id])
        if not groups:
            raise HTTPException(status_code=400, detail=f"Group with id={group_id} does not exist")

    owner_id, effective_group_id = compute_owner_and_group(vis, group_id, current_user["id"])

    doc_repo = SQLAlchemyDocumentRepository(db)
    doc = Document(filename=filename, visibility=vis, owner_id=owner_id, group_id=effective_group_id)
    saved_doc = doc_repo.save(doc)

    storage_key = f"uploads/public/{saved_doc.id}_{filename}"
    storage = get_storage()
    storage.upload_file(storage_key, data)
    doc_repo.set_source_path(saved_doc.id, storage_key)

    db.commit()

    background_tasks.add_task(
        _process_upload_background,
        document_id=saved_doc.id,
        filename=filename,
        visibility=visibility,
        owner_id=owner_id,
        group_id=effective_group_id,
        replace_id=None,
    )

    return UploadStatusResponse(status="processing", document_id=saved_doc.id, filename=filename)


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from presentation.api.dependencies import create_document_service
    service = create_document_service(db)
    return service.list_documents(current_user["id"], current_user["kind"])


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document_status(
    document_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from presentation.api.dependencies import create_document_service
    service = create_document_service(db)
    return service.get_document(document_id, current_user["id"], current_user["kind"], current_user["role"])


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from presentation.api.dependencies import create_document_service
    service = create_document_service(db)
    service.delete_document(document_id, current_user["id"], current_user["role"])
    return {"status": "deleted", "document_id": document_id}
