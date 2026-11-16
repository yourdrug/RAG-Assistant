"""Knowledge space endpoints — listing for every user, management for ADMIN."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Response

from application.ports.action_logger import ActionLoggerPort
from application.ports.rate_limit import RateLimitPolicyName
from application.spaces import KnowledgeSpaceService, SpaceChanges
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from presentation.api.auth_dependencies import get_current_user, require_admin
from presentation.api.dependencies import create_action_logger, create_knowledge_space_service
from presentation.api.rate_limit import rate_limit
from presentation.api.schemas import CurrentUser
from presentation.api.schemas.spaces import (
    AssignClientSpaceRequest,
    CreateKnowledgeSpaceRequest,
    DomainPacksResponse,
    KnowledgeSpaceResponse,
    UpdateKnowledgeSpaceRequest,
)


if TYPE_CHECKING:
    from application.spaces import KnowledgeSpaceDTO


router = APIRouter(prefix="/spaces", tags=["spaces"])


def admin_context(admin: CurrentUser) -> UserContext:
    return UserContext(user_id=admin.id, user_kind=UserKind(admin.kind), user_role=UserRole(admin.role))


def to_response(space: KnowledgeSpaceDTO) -> KnowledgeSpaceResponse:
    return KnowledgeSpaceResponse(
        id=space.id,
        name=space.name,
        pack_key=space.pack_key,
        can_upload=space.can_upload,
        upload_group_ids=list(space.upload_group_ids) if space.upload_group_ids is not None else None,
    )


@router.get("", response_model=list[KnowledgeSpaceResponse])
async def list_spaces(
    *,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[KnowledgeSpaceService, Depends(create_knowledge_space_service)],
) -> list[KnowledgeSpaceResponse]:
    """Spaces a question may be asked in; a space never grants access to documents."""
    spaces = await service.list_for_user(current_user.id, current_user.kind, current_user.role)
    return [to_response(space) for space in spaces]


@router.get("/packs", response_model=DomainPacksResponse)
async def list_domain_packs(
    *,
    _admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[KnowledgeSpaceService, Depends(create_knowledge_space_service)],
) -> DomainPacksResponse:
    """All registered domain packs, available to new spaces."""
    return DomainPacksResponse(pack_keys=list(service.pack_keys()))


@router.post(
    "",
    response_model=KnowledgeSpaceResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def create_space(
    *,
    req: CreateKnowledgeSpaceRequest,
    admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[KnowledgeSpaceService, Depends(create_knowledge_space_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> KnowledgeSpaceResponse:
    space = await service.create(admin_context(admin), req.name, req.pack_key, req.upload_group_ids)
    log("space.create", user_id=admin.id, details={"space_id": space.id, "pack_key": space.pack_key})
    return to_response(space)


@router.patch(
    "/{space_id}",
    response_model=KnowledgeSpaceResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def update_space(
    *,
    space_id: int,
    req: UpdateKnowledgeSpaceRequest,
    admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[KnowledgeSpaceService, Depends(create_knowledge_space_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> KnowledgeSpaceResponse:
    space = await service.update(
        admin_context(admin), space_id, SpaceChanges(name=req.name, upload_group_ids=req.upload_group_ids)
    )
    log("space.update", user_id=admin.id, details={"space_id": space_id})
    return to_response(space)


@router.delete(
    "/{space_id}",
    status_code=204,
    response_class=Response,
    response_model=None,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def delete_space(
    *,
    space_id: int,
    admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[KnowledgeSpaceService, Depends(create_knowledge_space_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> None:
    await service.delete(space_id)
    log("space.delete", user_id=admin.id, details={"space_id": space_id})


@router.put(
    "/clients/{client_id}",
    status_code=204,
    response_class=Response,
    response_model=None,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def assign_client_space(
    *,
    client_id: int,
    req: AssignClientSpaceRequest,
    admin: Annotated[CurrentUser, Depends(require_admin)],
    service: Annotated[KnowledgeSpaceService, Depends(create_knowledge_space_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> None:
    await service.assign_client(admin_context(admin), client_id, req.space_id)
    log("space.assign_client", user_id=admin.id, details={"client_id": client_id, "space_id": req.space_id})
