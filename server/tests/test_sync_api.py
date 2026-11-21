"""Sync channel API and GitHub webhook through the real ASGI application."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import httpx
from pydantic import SecretStr
import pytest
import pytest_asyncio

from application.sync import SyncChannelService, SyncRunRequests
from config import StorageSection, SyncSection
from domain.sync.run import SyncRunStatus, SyncTrigger
from main import app
from presentation.api import dependencies as deps
from presentation.api.auth_dependencies import get_current_user
from presentation.api.routes import sync as sync_routes
from presentation.api.schemas import CurrentUser
from tests.fakes import DEFAULT_SPACE_ID, FakeUnitOfWorkFactory, admit_all_users


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from tests.fakes import FakeUnitOfWork


SECRET = "webhook-secret"  # noqa: S105 - test signing key
ADMIN_ID, RESPONSIBLE_ID, OTHER_ID = 1, 2, 3


class Api:
    def __init__(self, client: httpx.AsyncClient, factory: FakeUnitOfWorkFactory) -> None:
        self.client = client
        self.factory = factory
        self.uow: FakeUnitOfWork = factory._uow

    def sign_in(self, user_id: int, role: str = "user") -> None:
        user = CurrentUser(
            id=user_id, email=f"u{user_id}@example.org", role=role, kind="internal", is_active=True
        )
        app.dependency_overrides[get_current_user] = lambda: user

    async def create_channel(self, folder: str = "docs") -> int:
        self.sign_in(ADMIN_ID, "admin")
        response = await self.client.post("/sync/channels", json=channel_body(folder=folder))
        assert response.status_code == 201, response.text
        return int(response.json()["id"])


def channel_body(**changes: object) -> dict[str, object]:
    return {
        "space_id": DEFAULT_SPACE_ID,
        "repository_id": 42,
        "repository": "acme/docs",
        "branch": "main",
        "folder": "docs",
        "exclusions": [],
        "visibility": "internal_public",
        "responsible_user_id": RESPONSIBLE_ID,
        **changes,
    }


def mock_dependency() -> MagicMock:
    """Zero-argument override: FastAPI would read a mock's parameters as query fields."""
    return MagicMock()


def sections(section: type[object]) -> object:
    if section is SyncSection:
        return SyncSection(github_instance="github.com", github_webhook_secret=SecretStr(SECRET))

    return StorageSection(s3_bucket="rag-documents")


@pytest_asyncio.fixture
async def api(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Api]:
    factory = FakeUnitOfWorkFactory()
    admit_all_users(factory._uow)

    for user_id in (ADMIN_ID, RESPONSIBLE_ID, OTHER_ID):
        factory._uow.fake_users.add_user(user_id)

    channels = SyncChannelService(factory)
    requests = SyncRunRequests(factory)
    old_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[deps.create_rate_limiter] = lambda: None
    app.dependency_overrides[deps.create_action_logger] = mock_dependency

    for auth_dependency in (
        deps.create_auth_service,
        deps.create_api_key_service,
        deps.create_api_key_provider,
    ):
        app.dependency_overrides[auth_dependency] = mock_dependency

    app.dependency_overrides[deps.create_sync_channel_service] = lambda: channels
    app.dependency_overrides[deps.create_sync_run_requests] = lambda: requests
    monkeypatch.setattr(sync_routes, "request_section", sections)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]  # Starlette and HTTPX disagree on the ASGI send annotation.
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield Api(client, factory)
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(old_overrides)


@pytest.mark.asyncio
async def test_channel_lifecycle_rights_on_every_endpoint(api: Api) -> None:
    api.sign_in(RESPONSIBLE_ID)
    assert (await api.client.post("/sync/channels", json=channel_body())).status_code == 409

    channel_id = await api.create_channel()
    created = (await api.client.get("/sync/channels")).json()
    assert [(c["id"], c["instance"], c["folder"]) for c in created] == [(channel_id, "github.com", "docs")]

    # The responsible user narrows and starts runs; an unrelated user sees and does nothing.
    api.sign_in(RESPONSIBLE_ID)
    narrowed = channel_body(folder="docs/api")
    del narrowed["space_id"]
    response = await api.client.put(f"/sync/channels/{channel_id}", json=narrowed)
    assert (response.status_code, response.json()["generation"]) == (200, 2)
    widened = {**narrowed, "folder": ""}
    assert (await api.client.put(f"/sync/channels/{channel_id}", json=widened)).status_code == 409
    run = await api.client.post(f"/sync/channels/{channel_id}/runs")
    assert (run.status_code, run.json()["status"], run.json()["trigger"]) == (202, "queued", "manual")
    assert len((await api.client.get(f"/sync/channels/{channel_id}/runs")).json()) == 1
    assert (await api.client.delete(f"/sync/channels/{channel_id}")).status_code == 409

    api.sign_in(OTHER_ID)
    assert (await api.client.get("/sync/channels")).json() == []
    assert (await api.client.post(f"/sync/channels/{channel_id}/runs")).status_code == 409
    assert (await api.client.get(f"/sync/channels/{channel_id}/runs")).status_code == 409
    assert (await api.client.put(f"/sync/channels/{channel_id}", json=narrowed)).status_code == 409

    api.sign_in(ADMIN_ID, "admin")
    deleted = await api.client.delete(f"/sync/channels/{channel_id}", params={"mode": "unlink"})
    assert deleted.json() == {"mode": "unlink", "documents": 0}


@pytest.mark.asyncio
async def test_channel_endpoints_require_authentication(api: Api) -> None:
    app.dependency_overrides.pop(get_current_user, None)
    assert (await api.client.get("/sync/channels")).status_code == 401


def signed(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


async def deliver(
    api: Api, payload: object, *, delivery: str = "d-1", signature: str | None = None
) -> httpx.Response:
    body = json.dumps(payload).encode()
    headers = {"X-GitHub-Event": "push", "X-GitHub-Delivery": delivery}

    if signature != "":
        headers["X-Hub-Signature-256"] = signature or signed(body)

    return await api.client.post("/sync/github/webhook", content=body, headers=headers)


PUSH = {"ref": "refs/heads/main", "repository": {"id": 42, "full_name": "acme/docs"}}


@pytest.mark.asyncio
async def test_unsigned_or_wrongly_signed_webhook_is_rejected(api: Api) -> None:
    await api.create_channel()
    assert (await deliver(api, PUSH, signature="")).status_code == 401
    assert (await deliver(api, PUSH, signature=signed(b"other body"))).status_code == 401
    assert (await deliver(api, PUSH, signature=signed(json.dumps(PUSH).encode(), "wrong"))).status_code == 401
    assert api.uow.fake_sync_runs.runs == {}


@pytest.mark.asyncio
async def test_push_queues_one_run_and_redelivery_does_not_restart_it(api: Api) -> None:
    channel_id = await api.create_channel()
    first = await deliver(api, PUSH)
    assert (first.status_code, first.json()) == (202, {"queued": 1})
    assert (await deliver(api, PUSH)).json() == {"queued": 0}
    [run] = api.uow.fake_sync_runs.runs.values()
    assert (run.channel_id, run.trigger, run.status, run.delivery_id) == (
        channel_id,
        SyncTrigger.WEBHOOK,
        SyncRunStatus.QUEUED,
        "d-1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {**PUSH, "ref": "refs/heads/feature"},
        {**PUSH, "ref": "refs/tags/v1"},
        {**PUSH, "repository": {"id": 7}},
        {"zen": "ping payload"},
    ],
    ids=["untracked-branch", "tag", "other-repository", "malformed"],
)
async def test_irrelevant_pushes_are_ignored(api: Api, payload: object) -> None:
    await api.create_channel()
    assert (await deliver(api, payload)).json() == {"queued": 0}
    assert api.uow.fake_sync_runs.runs == {}


@pytest.mark.asyncio
async def test_webhook_is_disabled_without_a_secret(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sync_routes,
        "request_section",
        lambda section: SyncSection() if section is SyncSection else sections(section),
    )
    assert (await deliver(api, PUSH)).status_code == 404


S3_CHANNEL = {
    "space_id": DEFAULT_SPACE_ID,
    "storage_kind": "s3",
    "folder": "docs",
    "visibility": "internal_public",
    "responsible_user_id": RESPONSIBLE_ID,
}


@pytest.mark.asyncio
async def test_s3_channel_uses_the_deployment_bucket_and_takes_no_repository_fields(api: Api) -> None:
    api.sign_in(ADMIN_ID, "admin")
    created = await api.client.post("/sync/channels", json=S3_CHANNEL)
    assert created.status_code == 201, created.text
    assert (created.json()["storage_kind"], created.json()["bucket"], created.json()["repository"]) == (
        "s3",
        "rag-documents",
        None,
    )
    mixed = await api.client.post("/sync/channels", json={**S3_CHANNEL, "folder": "guides", "branch": "main"})
    assert mixed.status_code == 422
    github_without_repository = await api.client.post(
        "/sync/channels", json={**S3_CHANNEL, "storage_kind": "github", "folder": "guides"}
    )
    assert github_without_repository.status_code == 422

    channel_id = created.json()["id"]
    api.sign_in(RESPONSIBLE_ID)
    assert (await api.client.post(f"/sync/channels/{channel_id}/adopt")).status_code == 409
    api.sign_in(ADMIN_ID, "admin")
    adopted = await api.client.post(f"/sync/channels/{channel_id}/adopt")
    assert adopted.json() == {"adopted": [], "different_access": [], "skipped": []}
