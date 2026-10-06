"""Real application ASGI contracts, keeping auth rules, middleware and handlers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from application.services.conversation_service import ConversationService
from domain.entities.message import Message
from domain.value_objects.message_role import MessageRole
from domain.value_objects.roles import UserRole
from main import app
from presentation.api.auth_dependencies import get_current_user
from presentation.api import dependencies as deps
from presentation.api.schemas import CurrentUser
from fakes import FakeUnitOfWorkFactory


@pytest_asyncio.fixture
async def asgi_client():
    old_overrides = dict(app.dependency_overrides)
    old_container = getattr(app.state, "container", None)
    app.state.container = None
    app.dependency_overrides[deps.create_rate_limiter] = lambda: None
    app.dependency_overrides[deps.create_action_logger] = lambda: MagicMock()
    app.dependency_overrides[deps.create_auth_service] = lambda: MagicMock()
    app.dependency_overrides[deps.create_api_key_provider] = lambda: MagicMock()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(old_overrides)
        app.state.container = old_container


def sign_in(role="user", user_id=1):
    user = CurrentUser(id=user_id, email="asgi@example.org", role=role, kind="internal", is_active=True)
    app.dependency_overrides[get_current_user] = lambda: user


@pytest.mark.asyncio
@pytest.mark.parametrize("dataset", [None, "custom"])
async def test_benchmark_launch_enqueues_database_dataset(asgi_client, tmp_path, dataset):
    from domain.value_objects.benchmark_dataset import BenchmarkDataset

    sign_in(role=UserRole.ADMIN.value)
    enqueuer = SimpleNamespace(enqueue_benchmark=AsyncMock())
    app.dependency_overrides[deps.create_job_enqueuer] = lambda: enqueuer
    app.dependency_overrides[deps.create_job_service] = lambda: SimpleNamespace(
        create_job=AsyncMock(return_value=8)
    )
    app.dependency_overrides[deps.create_benchmark_config] = lambda: SimpleNamespace(
        data_dir=str(tmp_path), retriever_top_k=7, llm_model="judge"
    )
    app.dependency_overrides[deps.create_idempotency_store] = lambda: None
    response = await asgi_client.post("/benchmark", json={} if dataset is None else {"dataset": dataset})
    assert response.status_code == 200
    enqueuer.enqueue_benchmark.assert_awaited_once_with(
        dataset=dataset or BenchmarkDataset.MAIN.value,
        out_dir=str(tmp_path / "benchmark_results"),
        top_k=7,
        judge_model="judge",
        job_id=8,
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_real_auth_required(asgi_client):
    response = await asgi_client.get("/conversations")
    assert response.status_code == 401
    assert "message" in response.json()
    assert response.headers["X-Request-ID"]


@pytest.mark.asyncio
@pytest.mark.parametrize("role,status", [("user", 403), ("curator", 403), ("admin", 200)])
async def test_benchmark_compare_route_and_roles(asgi_client, role, status):
    sign_in(role)
    service = SimpleNamespace(compare=AsyncMock(return_value=([], {})))
    app.dependency_overrides[deps.create_benchmark_run_service] = lambda: service
    response = await asgi_client.get("/admin/benchmark/runs/compare?ids=1,2")
    assert response.status_code == status
    if status == 200:
        assert response.json() == {"runs": [], "diff": {}}
        service.compare.assert_awaited_once_with([1, 2])
    else:
        service.compare.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("filename,data", [("bad.exe", b"text"), ("bad.pdf", b"%PDF-broken")])
async def test_file_validation_returns_422_before_upload(asgi_client, filename, data):
    sign_in("admin")
    command = SimpleNamespace(upload=AsyncMock())
    app.dependency_overrides[deps.create_document_command_service] = lambda: command
    app.dependency_overrides[deps.create_job_service] = lambda: MagicMock()
    app.dependency_overrides[deps.create_job_enqueuer] = lambda: MagicMock()
    app.dependency_overrides[deps.create_upload_config] = lambda: SimpleNamespace(max_upload_size_mb=1)
    app.dependency_overrides[deps.create_idempotency_store] = lambda: MagicMock()
    response = await asgi_client.post(
        "/documents", files={"file": (filename, data)}, data={"visibility": "internal_public"}
    )
    assert response.status_code == 422
    assert "file" in response.json()["errors"]
    command.upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_conversation_cursor_and_ownership(asgi_client):
    factory = FakeUnitOfWorkFactory()
    factory._uow.conversations._convs[7] = {"id": 7, "user_id": 1, "summary": None}
    factory._uow.messages._messages = [
        Message(id=i, conversation_id=7, role=MessageRole.USER, content=str(i)) for i in range(1, 106)
    ]
    service = ConversationService(factory, MagicMock(), MagicMock())
    app.dependency_overrides[deps.create_conversation_service] = lambda: service
    sign_in(user_id=1)
    response = await asgi_client.get("/conversations/7")
    assert response.status_code == 200
    assert len(response.json()["messages"]) == 100
    assert response.json()["next_cursor"] == 6
    older = await asgi_client.get("/conversations/7?before_id=6")
    assert [m["id"] for m in older.json()["messages"]] == [1, 2, 3, 4, 5]
    assert older.json()["next_cursor"] is None
    sign_in(user_id=2)
    assert (await asgi_client.get("/conversations/7")).status_code == 403
    sign_in("admin", 2)
    assert (await asgi_client.get("/conversations/7")).status_code == 200


@pytest.mark.asyncio
async def test_chat_outage_preserves_retry_after(asgi_client):
    from domain.exceptions import LLMUnavailableError

    sign_in()
    service = SimpleNamespace(sync_chat=AsyncMock(side_effect=LLMUnavailableError()))
    app.dependency_overrides[deps.create_chat_service] = lambda: service
    response = await asgi_client.post("/chat/sync", json={"question": "test"})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"
    assert "message" in response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id,role,status", [(1, "user", 200), (2, "user", 409), (2, "admin", 200)])
async def test_private_document_ownership_through_real_route(asgi_client, user_id, role, status):
    from application.services.document_query_service import DocumentQueryService
    from domain.entities.document import Document
    from domain.value_objects.visibility import DocumentVisibility

    factory = FakeUnitOfWorkFactory()
    document = await factory._uow.documents.save(
        Document(filename="private.pdf", owner_id=1, visibility=DocumentVisibility.INTERNAL_PRIVATE)
    )
    service = DocumentQueryService(factory)
    app.dependency_overrides[deps.create_document_query_service] = lambda: service
    sign_in(role, user_id)
    response = await asgi_client.get(f"/documents/{document.id}")
    assert response.status_code == status
    if status == 200:
        assert response.json()["filename"] == "private.pdf"
    else:
        assert "private.pdf" not in response.text
