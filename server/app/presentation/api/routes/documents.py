"""Document endpoints — thin wrappers around DocumentService."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from infrastructure.auth.fastapi_dependencies import get_current_user
from infrastructure.database.session import get_db
from sqlalchemy.orm import Session

from presentation.api.dependencies import create_document_service
from presentation.api.schemas import DocumentResponse

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=DocumentResponse)
async def upload_document(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    visibility: str = Form(...),
    group_id: int | None = Form(None),
):
    service = create_document_service(db)
    data = await file.read()
    result = await service.upload(
        filename=file.filename,
        file_data=data,
        visibility=visibility,
        group_id=group_id,
        user_id=current_user["id"],
        user_kind=current_user["kind"],
        user_role=current_user["role"],
    )
    return result


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = create_document_service(db)
    return service.list_documents(current_user["id"], current_user["kind"])


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document_status(
    document_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = create_document_service(db)
    return service.get_document(document_id, current_user["id"], current_user["kind"], current_user["role"])


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    service = create_document_service(db)
    service.delete_document(document_id, current_user["id"], current_user["role"])
    return {"status": "deleted", "document_id": document_id}
