"""Sync channel endpoints and the GitHub App webhook.

Channels: an administrator creates, widens and deletes them; the responsible user may
narrow a channel, start a run and read its history. The webhook only says which branch
changed: its body is trusted for nothing else, and each run reads the storage itself.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

from application.ports.rate_limit import RateLimitPolicyName
from application.sync import SyncChannelService, SyncRunRequests
from config import StorageSection, SyncSection, request_section
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.sync.channel import (
    ChannelAudience,
    ChannelDeletionMode,
    ChannelSettings,
    GitHubLocation,
    S3Location,
    StorageKind,
)
from domain.sync.run import FileOutcomeKind, SyncTrigger
from presentation.api.auth_dependencies import get_current_user
from presentation.api.dependencies import create_sync_channel_service, create_sync_run_requests
from presentation.api.rate_limit import rate_limit
from presentation.api.schemas import CurrentUser
from presentation.api.schemas.sync import (
    AdoptionResponse,
    ChannelSettingsRequest,
    CreateChannelRequest,
    DeleteChannelResponse,
    FileOutcomeResponse,
    SkippedPathResponse,
    StorageKindsResponse,
    SyncChannelResponse,
    SyncRunResponse,
    WebhookResponse,
)


if TYPE_CHECKING:
    from application.sync.channels import ChannelView
    from domain.sync.channel import StorageLocation, SyncChannel
    from domain.sync.run import SyncRun


log = logging.getLogger("default")
router = APIRouter(prefix="/sync", tags=["sync"])

PROBLEM_KINDS = frozenset({FileOutcomeKind.FAILED, FileOutcomeKind.CONFLICT})
SIGNATURE_PREFIX = "sha256="
BRANCH_REF_PREFIX = "refs/heads/"
# A push delivery's body is small; larger bodies are rejected before being parsed.
MAX_WEBHOOK_BYTES = 25 * 1024 * 1024


def actor(user: CurrentUser) -> UserContext:
    return UserContext(user_id=user.id, user_kind=UserKind(user.kind), user_role=UserRole(user.role))


def to_location(request: ChannelSettingsRequest) -> StorageLocation:
    if request.storage_kind == StorageKind.S3:
        return S3Location(request_section(StorageSection).s3_bucket)

    return GitHubLocation(
        request_section(SyncSection).github_instance,
        request.repository_id or 0,
        request.repository or "",
        request.branch or "",
    )


def to_settings(request: ChannelSettingsRequest) -> ChannelSettings:
    return ChannelSettings(
        location=to_location(request),
        audience=ChannelAudience(request.visibility, request.owner_id, request.group_id),
        responsible_user_id=request.responsible_user_id,
        folder=request.folder,
        exclusions=tuple(request.exclusions),
    )


def run_response(run: SyncRun) -> SyncRunResponse:
    return SyncRunResponse(
        id=run.id or 0,
        status=run.status,
        trigger=run.trigger,
        storage_revision=run.storage_revision,
        error=run.error,
        counts=run.counts(),
        problems=[
            FileOutcomeResponse(path=item.path, kind=item.kind, detail=item.detail)
            for item in run.outcomes
            if item.kind in PROBLEM_KINDS
        ],
        creation_date=run.creation_date,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


def channel_response(channel: SyncChannel, last_run: SyncRun | None = None) -> SyncChannelResponse:
    settings = channel.settings
    github = settings.location if isinstance(settings.location, GitHubLocation) else None
    return SyncChannelResponse(
        id=channel.id or 0,
        space_id=channel.space_id,
        storage_kind=channel.storage_kind,
        instance=github.instance if github else None,
        repository_id=github.repository_id if github else None,
        repository=github.repository if github else None,
        branch=github.branch if github else None,
        bucket=settings.location.bucket if isinstance(settings.location, S3Location) else None,
        folder=settings.folder,
        exclusions=list(settings.exclusions),
        visibility=settings.audience.visibility,
        owner_id=settings.audience.owner_id,
        group_id=settings.audience.group_id,
        responsible_user_id=settings.responsible_user_id,
        generation=channel.generation,
        last_run=run_response(last_run) if last_run is not None else None,
    )


def view_response(view: ChannelView) -> SyncChannelResponse:
    return channel_response(view.channel, view.last_run)


@router.get("/storage-kinds", response_model=StorageKindsResponse)
async def storage_kinds(
    *,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[SyncChannelService, Depends(create_sync_channel_service)],
) -> StorageKindsResponse:
    """Kinds with a configured storage handler; a channel of another kind is refused."""
    del current_user
    return StorageKindsResponse(kinds=sorted(service.storage_kinds))


@router.get("/channels", response_model=list[SyncChannelResponse])
async def list_channels(
    *,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[SyncChannelService, Depends(create_sync_channel_service)],
    space_id: Annotated[int | None, Query(gt=0)] = None,
) -> list[SyncChannelResponse]:
    """Administrators see every channel; other users those they are responsible for."""
    return [view_response(view) for view in await service.list_visible(actor(current_user), space_id)]


@router.post(
    "/channels",
    response_model=SyncChannelResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def create_channel(
    *,
    req: CreateChannelRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[SyncChannelService, Depends(create_sync_channel_service)],
) -> SyncChannelResponse:
    """Bind a repository folder to a space (administrators only)."""
    channel = await service.create(req.space_id, to_settings(req), actor(current_user))
    return channel_response(channel)


@router.put(
    "/channels/{channel_id}",
    response_model=SyncChannelResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def change_channel(
    *,
    channel_id: int,
    req: ChannelSettingsRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[SyncChannelService, Depends(create_sync_channel_service)],
) -> SyncChannelResponse:
    """Change the scope; only an administrator may widen it."""
    channel = await service.change(channel_id, to_settings(req), actor(current_user))
    return channel_response(channel)


@router.delete(
    "/channels/{channel_id}",
    response_model=DeleteChannelResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def delete_channel(
    *,
    channel_id: int,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[SyncChannelService, Depends(create_sync_channel_service)],
    mode: ChannelDeletionMode = ChannelDeletionMode.DELETE_DOCUMENTS,
) -> DeleteChannelResponse:
    """Delete the channel with its documents, or keep them as manual documents."""
    affected = await service.delete(channel_id, mode, actor(current_user))
    return DeleteChannelResponse(mode=mode, documents=affected)


@router.post(
    "/channels/{channel_id}/adopt",
    response_model=AdoptionResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def adopt_ingested(
    *,
    channel_id: int,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[SyncChannelService, Depends(create_sync_channel_service)],
) -> AdoptionResponse:
    """Move documents ingested from an S3 channel's prefix into the channel (administrators only)."""
    report = await service.adopt_ingested(channel_id, actor(current_user))
    return AdoptionResponse(
        adopted=report.adopted,
        different_access=report.different_access,
        skipped=[SkippedPathResponse(path=item.path, reason=item.reason) for item in report.skipped],
    )


@router.post(
    "/channels/{channel_id}/runs",
    response_model=SyncRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def start_run(
    *,
    channel_id: int,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    requests: Annotated[SyncRunRequests, Depends(create_sync_run_requests)],
) -> SyncRunResponse:
    """Queue a run now ("sync now"); it cancels an unfinished one."""
    run = await requests.request(channel_id, SyncTrigger.MANUAL, actor(current_user))

    if run is None:  # Only a redelivered webhook is deduplicated.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Run was not queued")

    return run_response(run)


@router.get("/channels/{channel_id}/runs", response_model=list[SyncRunResponse])
async def list_runs(
    *,
    channel_id: int,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[SyncChannelService, Depends(create_sync_channel_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[SyncRunResponse]:
    return [run_response(run) for run in await service.runs(channel_id, actor(current_user), limit)]


def signature_valid(secret: str, body: bytes, signature: str | None) -> bool:
    if not signature or not signature.startswith(SIGNATURE_PREFIX):
        return False

    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.removeprefix(SIGNATURE_PREFIX))


async def read_limited_body(request: Request, limit: int) -> bytes:
    """The body, refused with 413 as soon as it exceeds ``limit`` (never buffered beyond it)."""
    declared = request.headers.get("content-length", "")

    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Body too large")

    parts: list[bytes] = []
    size = 0

    async for part in request.stream():
        size += len(part)

        if size > limit:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Body too large")

        parts.append(part)

    return b"".join(parts)


def pushed_branch(payload: object) -> tuple[int, str] | None:
    """Repository id and branch of a push; ``None`` for tags or malformed bodies."""
    if not isinstance(payload, dict):
        return None

    ref, repository = payload.get("ref"), payload.get("repository")
    repository_id = repository.get("id") if isinstance(repository, dict) else None

    if not isinstance(ref, str) or not ref.startswith(BRANCH_REF_PREFIX) or type(repository_id) is not int:
        return None

    return repository_id, ref.removeprefix(BRANCH_REF_PREFIX)


@router.post("/github/webhook", response_model=WebhookResponse, status_code=status.HTTP_202_ACCEPTED)
async def github_webhook(
    *,
    request: Request,
    requests: Annotated[SyncRunRequests, Depends(create_sync_run_requests)],
    x_hub_signature_256: Annotated[str | None, Header()] = None,
    x_github_event: Annotated[str | None, Header()] = None,
    x_github_delivery: Annotated[str | None, Header(max_length=64)] = None,
) -> WebhookResponse:
    """Queue runs of the channels reading the pushed branch; answers before any run executes."""
    sync = request_section(SyncSection)
    secret = sync.github_webhook_secret.get_secret_value()

    if not secret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Webhook is not configured")

    body = await read_limited_body(request, MAX_WEBHOOK_BYTES)

    if not signature_valid(secret, body, x_hub_signature_256):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    if x_github_event != "push" or not x_github_delivery:
        return WebhookResponse(queued=0)

    try:
        pushed = pushed_branch(json.loads(body))
    except ValueError:
        pushed = None

    if pushed is None:
        return WebhookResponse(queued=0)

    repository_id, branch = pushed
    queued = await requests.request_for_push(sync.github_instance, repository_id, branch, x_github_delivery)
    return WebhookResponse(queued=queued)
