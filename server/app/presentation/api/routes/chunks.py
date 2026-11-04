"""Chunk endpoints — CRUD operations for document chunks."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from application.chunks import ChunkCursorQuery, ChunkMutationService, ChunkQueryService
from application.documents import ManualDocumentService
from application.ports.action_logger import ActionLoggerPort
from application.ports.rate_limit import RateLimitPolicyName
from domain.identity.capabilities import Capability
from domain.identity.roles import UserKind, UserRole
from domain.knowledge.enums import DocumentVisibility
from presentation.api.auth_dependencies import get_current_user, require_capability
from presentation.api.dependencies import (
    create_action_logger,
    create_chunk_mutation_service,
    create_chunk_query_service,
    create_manual_document_service,
)
from presentation.api.rate_limit import rate_limit
from presentation.api.schemas import (
    ChunkCreateRequest,
    ChunkCursorListResponse,
    ChunkDeleteResponse,
    ChunkEditRequest,
    ChunkListResponse,
    ChunkResponse,
    CurrentUser,
    DocumentResponse,
    ManualDocumentRequest,
)


logger = logging.getLogger("default")

router = APIRouter(tags=["chunks"])


@router.get("/documents/{document_id}/chunks", response_model=ChunkListResponse)
async def list_chunks(
    *,
    document_id: int,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    highlight: Annotated[str | None, Query(description="Comma-separated content_hashes to filter")] = None,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    chunk_service: Annotated[ChunkQueryService, Depends(create_chunk_query_service)],
) -> ChunkListResponse:
    """List chunks for a document. When highlight is set, only matching chunks are returned."""
    content_hashes = [h.strip() for h in highlight.split(",") if h.strip()] if highlight else None
    chunks, total = await chunk_service.list_chunks(
        document_id=document_id,
        user_id=current_user.id,
        user_kind=UserKind(current_user.kind),
        user_role=UserRole(current_user.role),
        limit=limit,
        offset=offset,
        content_hashes=content_hashes,
    )
    chunk_responses = [
        ChunkResponse(
            id=c.id,
            document_id=c.document_id,
            chunk_index=c.chunk_index,
            content=c.content,
            filename=c.filename,
            visibility=c.visibility,
            doc_domain=c.doc_domain,
            owner_id=c.owner_id,
            group_id=c.group_id,
            edited_at=c.edited_at,
            edited_by=c.edited_by,
            manual=c.manual,
            creation_date=c.creation_date,
            content_hash=c.content_hash,
            section=c.section,
            heading=c.heading,
            heading_level=c.heading_level,
            content_type=c.content_type,
            doc_title=c.doc_title,
            doc_type=c.doc_type,
        )
        for c in chunks
    ]
    return ChunkListResponse(
        chunks=chunk_responses,
        total=total,
        document_id=document_id,
    )


@router.get("/documents/{document_id}/chunks/cursor", response_model=ChunkCursorListResponse)
async def list_chunks_cursor(
    *,
    document_id: int,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query(description="Opaque cursor from previous response")] = None,
    direction: Annotated[str, Query(pattern="^(next|prev)$")] = "next",
    highlight: Annotated[str | None, Query(description="Comma-separated content_hashes to filter")] = None,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    chunk_service: Annotated[ChunkQueryService, Depends(create_chunk_query_service)],
) -> ChunkCursorListResponse:
    """List chunks with cursor-based (keyset) pagination."""
    content_hashes = [h.strip() for h in highlight.split(",") if h.strip()] if highlight else None
    page = await chunk_service.list_chunks_cursor(
        document_id=document_id,
        user_id=current_user.id,
        user_kind=UserKind(current_user.kind),
        user_role=UserRole(current_user.role),
        query=ChunkCursorQuery(
            limit=limit,
            cursor=cursor,
            direction=direction,
            content_hashes=content_hashes,
        ),
    )
    chunk_responses = [
        ChunkResponse(
            id=c.id,
            document_id=c.document_id,
            chunk_index=c.chunk_index,
            content=c.content,
            filename=c.filename,
            visibility=c.visibility,
            doc_domain=c.doc_domain,
            owner_id=c.owner_id,
            group_id=c.group_id,
            edited_at=c.edited_at,
            edited_by=c.edited_by,
            manual=c.manual,
            creation_date=c.creation_date,
            content_hash=c.content_hash,
            section=c.section,
            heading=c.heading,
            heading_level=c.heading_level,
            content_type=c.content_type,
            doc_title=c.doc_title,
            doc_type=c.doc_type,
        )
        for c in page.items
    ]
    return ChunkCursorListResponse(
        chunks=chunk_responses,
        next_cursor=page.next_cursor,
        prev_cursor=page.prev_cursor,
        document_id=document_id,
    )


@router.post(
    "/documents/{document_id}/chunks",
    response_model=ChunkResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def add_chunk(
    *,
    document_id: int,
    request: ChunkCreateRequest,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.CHUNKS_MANAGE))],
    chunk_service: Annotated[ChunkMutationService, Depends(create_chunk_mutation_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> ChunkResponse:
    """Add a new chunk to an existing document."""
    result = await chunk_service.add_chunk(
        document_id=document_id,
        content=request.content,
        user_id=current_user.id,
        user_role=UserRole(current_user.role),
        page=request.page,
        section=request.section,
    )

    log(
        "chunk.create",
        user_id=current_user.id,
        details={"document_id": document_id, "chunk_id": result.id},
    )

    return ChunkResponse(
        id=result.id,
        document_id=result.document_id,
        chunk_index=result.chunk_index,
        content=result.content,
        manual=result.manual,
        warning=result.warning,
    )


@router.put(
    "/documents/{document_id}/chunks/{chunk_id}",
    response_model=ChunkResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def edit_chunk(
    *,
    document_id: int,
    chunk_id: int,
    request: ChunkEditRequest,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.CHUNKS_MANAGE))],
    chunk_service: Annotated[ChunkMutationService, Depends(create_chunk_mutation_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> ChunkResponse:
    """Edit an existing chunk's content with automatic re-embedding."""
    result = await chunk_service.edit_chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        content=request.content,
        user_id=current_user.id,
        user_role=UserRole(current_user.role),
    )

    log(
        "chunk.edit",
        user_id=current_user.id,
        details={"document_id": document_id, "chunk_id": chunk_id},
    )

    return ChunkResponse(
        id=result.id,
        document_id=result.document_id,
        chunk_index=result.chunk_index,
        content=result.content,
        edited_at=result.edited_at,
        edited_by=result.edited_by,
        manual=result.manual,
        warning=result.warning,
    )


@router.delete(
    "/documents/{document_id}/chunks/{chunk_id}",
    response_model=ChunkDeleteResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def delete_chunk(
    *,
    document_id: int,
    chunk_id: int,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.CHUNKS_MANAGE))],
    chunk_service: Annotated[ChunkMutationService, Depends(create_chunk_mutation_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> ChunkDeleteResponse:
    """Delete a single chunk."""
    await chunk_service.delete_chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        user_id=current_user.id,
        user_role=UserRole(current_user.role),
    )

    log(
        "chunk.delete",
        user_id=current_user.id,
        details={"document_id": document_id, "chunk_id": chunk_id},
    )

    return ChunkDeleteResponse(status="deleted", chunk_id=chunk_id)


@router.post(
    "/documents/manual",
    response_model=DocumentResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def create_manual_document(
    *,
    request: ManualDocumentRequest,
    current_user: Annotated[CurrentUser, Depends(require_capability(Capability.CHUNKS_MANAGE))],
    document_service: Annotated[ManualDocumentService, Depends(create_manual_document_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> DocumentResponse:
    """Create a virtual document container for manual chunks."""
    result = await document_service.create_manual_document(
        title=request.title,
        visibility=DocumentVisibility.parse(request.visibility),
        user_id=current_user.id,
        user_kind=UserKind(current_user.kind),
        user_role=UserRole(current_user.role),
        group_id=request.group_id,
    )

    log(
        "document.create_manual",
        user_id=current_user.id,
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
