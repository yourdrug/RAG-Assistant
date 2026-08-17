"""Chunk endpoints — CRUD operations for document chunks."""

from __future__ import annotations

import logging

from application.services.chunk_service import ChunkService
from fastapi import APIRouter, Depends, Query
from infrastructure.logging.actions import log_action

from presentation.api.auth_dependencies import get_current_user
from presentation.api.dependencies import create_chunk_service
from presentation.api.schemas import (
    ChunkCreateRequest,
    ChunkEditRequest,
    ChunkListResponse,
    ChunkResponse,
    DocumentResponse,
    ManualDocumentRequest,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["chunks"])


@router.get("/documents/{document_id}/chunks", response_model=ChunkListResponse)
async def list_chunks(
    document_id: int,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user),
    chunk_service: ChunkService = Depends(create_chunk_service),
):
    """List all chunks for a document with pagination."""
    chunks, total = await chunk_service.list_chunks(
        document_id=document_id,
        user_id=current_user["id"],
        user_kind=current_user["kind"],
        user_role=current_user["role"],
        limit=limit,
        offset=offset,
    )
    return ChunkListResponse(chunks=chunks, total=total, document_id=document_id)


@router.post("/documents/{document_id}/chunks", response_model=ChunkResponse)
async def add_chunk(
    document_id: int,
    request: ChunkCreateRequest,
    current_user: dict = Depends(get_current_user),
    chunk_service: ChunkService = Depends(create_chunk_service),
):
    """Add a new chunk to an existing document."""
    result = await chunk_service.add_chunk(
        document_id=document_id,
        content=request.content,
        user_id=current_user["id"],
        user_role=current_user["role"],
        page=request.page,
        section=request.section,
    )

    log_action(
        "chunk.create",
        user_id=current_user["id"],
        details={"document_id": document_id, "chunk_id": result["id"]},
    )

    return ChunkResponse(**result)


@router.put("/documents/{document_id}/chunks/{chunk_id}", response_model=ChunkResponse)
async def edit_chunk(
    document_id: int,
    chunk_id: int,
    request: ChunkEditRequest,
    current_user: dict = Depends(get_current_user),
    chunk_service: ChunkService = Depends(create_chunk_service),
):
    """Edit an existing chunk's content with automatic re-embedding."""
    result = await chunk_service.edit_chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        content=request.content,
        user_id=current_user["id"],
        user_role=current_user["role"],
    )

    log_action(
        "chunk.edit",
        user_id=current_user["id"],
        details={"document_id": document_id, "chunk_id": chunk_id},
    )

    return ChunkResponse(**result)


@router.delete("/documents/{document_id}/chunks/{chunk_id}")
async def delete_chunk(
    document_id: int,
    chunk_id: int,
    current_user: dict = Depends(get_current_user),
    chunk_service: ChunkService = Depends(create_chunk_service),
):
    """Delete a single chunk."""
    await chunk_service.delete_chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        user_id=current_user["id"],
        user_role=current_user["role"],
    )

    log_action(
        "chunk.delete",
        user_id=current_user["id"],
        details={"document_id": document_id, "chunk_id": chunk_id},
    )

    return {"status": "deleted", "chunk_id": chunk_id}


@router.post("/documents/manual", response_model=DocumentResponse)
async def create_manual_document(
    request: ManualDocumentRequest,
    current_user: dict = Depends(get_current_user),
    chunk_service: ChunkService = Depends(create_chunk_service),
):
    """Create a virtual document container for manual chunks."""
    result = await chunk_service.create_manual_document(
        title=request.title,
        visibility=request.visibility,
        user_id=current_user["id"],
        user_kind=current_user["kind"],
        user_role=current_user["role"],
        group_id=request.group_id,
    )

    log_action(
        "document.create_manual",
        user_id=current_user["id"],
        details={"document_id": result.id, "title": request.title},
    )

    return DocumentResponse(
        id=result.id,
        filename=result.filename,
        source_path=result.source_path or "",
        visibility=result.visibility,
        owner_id=result.owner_id,
        group_id=result.group_id,
        status=result.status,
        error_message=result.error_message,
        warning_message=result.warning_message,
        quality_score=result.quality_score,
        chunks=result.chunks,
        chars=result.chars,
        creation_date=result.creation_date,
        indexed_at=result.indexed_at,
        source_type=result.source_type,
        has_manual_edits=result.has_manual_edits,
    )
