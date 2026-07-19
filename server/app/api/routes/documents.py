"""
api/routes/documents.py — Document endpoints. Thin wrappers around DocumentService.
"""

from fastapi import APIRouter, Depends, File, Form, UploadFile
from infrastructure.auth import get_current_user
from infrastructure.database import get_db
from services.document_service import DocumentService
from sqlalchemy.orm import Session

from api.schemas import DocumentResponse

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=DocumentResponse)
async def upload_document(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    visibility: str = Form(...),
    group_id: int | None = Form(None),
):
    return await DocumentService().upload_document(file, visibility, group_id, current_user, db)


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return DocumentService().list_documents(current_user, db)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document_status(
    document_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return DocumentService().get_document(document_id, current_user, db)


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    DocumentService().delete_document(document_id, current_user, db)
    return {"status": "deleted", "document_id": document_id}
