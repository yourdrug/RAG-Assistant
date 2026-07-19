from config import settings
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from infrastructure.acl import can_view_document, owner_and_group_for, validate_visibility
from infrastructure.auth import get_current_user
from infrastructure.database import (
    create_document_row,
    delete_document_row,
    find_active_slot,
    get_db,
    get_document,
    list_documents_visible,
    set_document_source_path,
)
from infrastructure.document_service import process_upload, save_upload
from sqlalchemy.orm import Session

from api.schemas import DocumentResponse

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=DocumentResponse)
async def upload_document(
    background_tasks,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    visibility: str = Form(...),
    group_id: int | None = Form(None),
):
    validate_visibility(visibility, group_id, current_user, db)
    owner_id, effective_group_id = owner_and_group_for(visibility, group_id, current_user)

    existing = find_active_slot(db, owner_id, file.filename, effective_group_id)
    if existing and existing["status"] in ("pending", "processing"):
        raise HTTPException(status_code=409, detail="Этот документ уже обрабатывается")
    replace_id = existing["id"] if existing and existing["status"] == "done" else None

    document_id = create_document_row(
        db, filename=file.filename, visibility=visibility, owner_id=owner_id, group_id=effective_group_id
    )

    storage_key = await save_upload(file, document_id, owner_id, effective_group_id)
    set_document_source_path(db, document_id, storage_key)

    background_tasks.add_task(
        process_upload,
        document_id,
        storage_key,
        file.filename,
        visibility,
        owner_id,
        effective_group_id,
        replace_id,
    )

    return get_document(db, document_id)


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return list_documents_visible(db, current_user)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document_status(
    document_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = get_document(db, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Документ не найден")
    if not can_view_document(db, current_user, doc):
        raise HTTPException(status_code=403, detail="Нет доступа к этому документу")
    return doc


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from infrastructure.storage import get_storage
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    doc = get_document(db, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Документ не найден")

    is_owner = doc["owner_id"] == current_user["id"]
    if not is_owner and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Можно удалять только свои документы")

    client = QdrantClient(url=settings.qdrant_url)
    client.delete(
        collection_name=settings.collection_name,
        points_selector=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]),
    )
    if doc["source_path"]:
        get_storage().delete_file(doc["source_path"])
    delete_document_row(db, document_id)
    return {"status": "deleted", "document_id": document_id}
