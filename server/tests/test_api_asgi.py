"""Real application ASGI contracts, keeping auth rules, middleware and handlers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from application.chat import ConversationService
from application.chunks import AddChunkResult, EditChunkResult
from domain.shared.pagination import CursorPage
from test_chunk_service import _make_chunk, _make_doc, _make_service, _make_uow
from application.documents import DocumentDTO
from domain.entities.document import Document
from domain.value_objects.visibility import DocumentVisibility
from domain.value_objects.source_type import SourceType
from domain.entities.message import Message
from domain.value_objects.message_role import MessageRole
from domain.value_objects.job_status import BackgroundJobStatus
from domain.identity.roles import UserRole
from domain.value_objects.doc_domain import DocDomain
from domain.repositories.chunk_repository import ChunkSearchResult
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
    app.dependency_overrides[deps.create_api_key_service] = lambda: MagicMock()
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
@pytest.mark.parametrize("path", ["/ingest", "/ingest/file"])
@pytest.mark.parametrize("domain", ["legal_typo", "", "AUTO"])
async def test_ingestion_rejects_unknown_domain_before_creating_job(asgi_client, path, domain):
    sign_in(UserRole.ADMIN.value)
    jobs = SimpleNamespace(create_job=AsyncMock(return_value=8))
    app.dependency_overrides[deps.create_job_service] = lambda: jobs
    app.dependency_overrides[deps.create_job_enqueuer] = lambda: SimpleNamespace(
        enqueue_ingest=AsyncMock(), enqueue_ingest_file=AsyncMock()
    )
    app.dependency_overrides[deps.create_ingest_service] = lambda: SimpleNamespace(
        resolve_docs_dir=lambda value: value, resolve_ingest_target=lambda value: value
    )
    app.dependency_overrides[deps.create_idempotency_store] = lambda: None
    response = await asgi_client.post(path, params={"domain": domain, "file_path": "docs/test.pdf"})
    assert response.status_code == 422
    assert "domain" in response.json()["errors"]
    jobs.create_job.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/ingest", "/ingest/file"])
@pytest.mark.parametrize("domain", [*(item.value for item in DocDomain), "auto"])
async def test_ingestion_accepts_supported_and_automatic_domains(asgi_client, path, domain):
    sign_in(UserRole.ADMIN.value)
    app.dependency_overrides[deps.create_job_service] = lambda: SimpleNamespace(
        create_job=AsyncMock(return_value=8)
    )
    app.dependency_overrides[deps.create_job_enqueuer] = lambda: SimpleNamespace(
        enqueue_ingest=AsyncMock(), enqueue_ingest_file=AsyncMock()
    )
    app.dependency_overrides[deps.create_ingest_service] = lambda: SimpleNamespace(
        resolve_docs_dir=lambda value: value, resolve_ingest_target=lambda value: value
    )
    app.dependency_overrides[deps.create_idempotency_store] = lambda: None
    response = await asgi_client.post(path, params={"domain": domain, "file_path": "docs/test.pdf"})
    assert response.status_code == 200
    assert response.json()["status"] == "started"


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", ["legal_typo", "auto"])
async def test_preview_index_rejects_unknown_domain_before_loading_file(asgi_client, domain):
    sign_in(UserRole.ADMIN.value)
    cache = SimpleNamespace(get_bytes=AsyncMock(return_value=None))
    app.dependency_overrides[deps.create_preview_cache] = lambda: cache
    app.dependency_overrides[deps.create_document_command_service] = lambda: MagicMock()
    app.dependency_overrides[deps.create_job_service] = lambda: MagicMock()
    app.dependency_overrides[deps.create_job_enqueuer] = lambda: MagicMock()
    response = await asgi_client.post("/admin/documents/preview/test/index", data={"doc_domain": domain})
    assert response.status_code == 422
    assert "doc_domain" in response.json()["errors"]
    cache.get_bytes.assert_not_awaited()


@pytest.mark.asyncio
async def test_exact_search_reports_full_total_and_requested_page(asgi_client):
    sign_in()
    row = ChunkSearchResult(3, 10, "test.txt", "matching needle text", 3)
    service = SimpleNamespace(exact_search=AsyncMock(return_value=([row], 53)))
    app.dependency_overrides[deps.create_search_service] = lambda: service
    response = await asgi_client.post("/search/exact", json={"query": "needle", "limit": 2, "offset": 2})
    assert response.status_code == 200
    assert response.json()["total"] == 53
    assert response.json()["limit"] == 2
    assert response.json()["offset"] == 2
    assert [row["chunk_id"] for row in response.json()["results"]] == [3]
    assert service.exact_search.call_args.kwargs["offset"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("offset", [-1, 0.5, "bad"])
async def test_exact_search_rejects_invalid_offset(asgi_client, offset):
    sign_in()
    service = SimpleNamespace(exact_search=AsyncMock())
    app.dependency_overrides[deps.create_search_service] = lambda: service
    response = await asgi_client.post("/search/exact", json={"query": "needle", "offset": offset})
    assert response.status_code == 422
    service.exact_search.assert_not_awaited()


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
    from domain.shared.exceptions import LLMUnavailableError

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
    from application.documents import DocumentQueryService
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_id,role,expected", [(1, UserRole.CURATOR, 202), (2, UserRole.CURATOR, 409), (1, UserRole.USER, 403)]
)
async def test_document_delete_queues_only_after_authorization(asgi_client, user_id, role, expected):
    from application.documents import DocumentDeletionJobs
    from domain.entities.background_job import BackgroundJob
    from domain.value_objects.job_type import JobType
    from domain.value_objects.visibility import DocumentVisibility

    factory = FakeUnitOfWorkFactory()
    document = await factory._uow.documents.save(
        Document(
            filename="private.pdf",
            owner_id=1,
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
        )
    )
    jobs = factory._uow.background_jobs
    jobs.create = AsyncMock(
        return_value=BackgroundJob(id=42, related_id=document.id, job_type=JobType.DOCUMENT_DELETION)
    )
    enqueuer = SimpleNamespace(enqueue_document_deletion=AsyncMock())
    service = DocumentDeletionJobs(factory, enqueuer)
    app.dependency_overrides[deps.create_document_deletion_jobs] = lambda: service
    sign_in(role.value, user_id)
    response = await asgi_client.delete(f"/documents/{document.id}")
    assert response.status_code == expected
    assert await factory._uow.documents.get_by_id(document.id) == document
    if expected == 202:
        assert response.json() == {
            "status": BackgroundJobStatus.PENDING.value,
            "document_id": document.id,
            "job_id": 42,
        }
        enqueuer.enqueue_document_deletion.assert_awaited_once_with(
            document_id=document.id, user_id=user_id, job_id=42
        )
        jobs.get_by_id = AsyncMock(
            return_value=BackgroundJob(id=42, related_id=document.id, job_type=JobType.DOCUMENT_DELETION)
        )
        progress = await asgi_client.get(f"/documents/{document.id}/deletion/42")
        assert progress.status_code == 200
        assert progress.json()["status"] == BackgroundJobStatus.PENDING.value
        await factory._uow.documents.delete(document.id)
        assert (await asgi_client.get(f"/documents/{document.id}/deletion/42")).status_code == 404
    else:
        jobs.create.assert_not_awaited()
        enqueuer.enqueue_document_deletion.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path,payload,operation",
    [
        ("GET", "/documents/1/chunks", None, "list_chunks"),
        ("GET", "/documents/1/chunks/cursor?limit=1", None, "list_chunks_cursor"),
        ("POST", "/documents/1/chunks", {"content": "text"}, "add_chunk"),
        ("PUT", "/documents/1/chunks/10", {"content": "text"}, "edit_chunk"),
        ("DELETE", "/documents/1/chunks/10", None, "delete_chunk"),
        (
            "POST",
            "/documents/manual",
            {"title": "Manual", "visibility": "internal_private"},
            "create_manual_document",
        ),
    ],
)
async def test_chunk_routes_resolve_composed_services(asgi_client, method, path, payload, operation):
    query = SimpleNamespace(
        list_chunks=AsyncMock(return_value=([], 0)),
        list_chunks_cursor=AsyncMock(return_value=CursorPage(items=[], next_cursor=None, prev_cursor=None)),
    )
    mutation = SimpleNamespace(
        add_chunk=AsyncMock(return_value=AddChunkResult(10, 1, 0, "text", True)),
        edit_chunk=AsyncMock(return_value=EditChunkResult(10, 1, 0, "text", "2026-01-01T00:00:00", 1, False)),
        delete_chunk=AsyncMock(),
    )
    manual = SimpleNamespace(
        create_manual_document=AsyncMock(
            return_value=DocumentDTO(
                id=1,
                filename="Manual",
                visibility="internal_private",
                status="done",
                source_type=SourceType.MANUAL.value,
            )
        )
    )
    app.state.container = SimpleNamespace(
        application=SimpleNamespace(
            chunk_query_service=query,
            chunk_mutation_service=mutation,
            manual_document_service=manual,
        )
    )
    sign_in(UserRole.ADMIN.value)
    response = await asgi_client.request(method, path, json=payload)
    assert response.status_code == 200
    owner = (
        query
        if operation.startswith("list_")
        else manual
        if operation == "create_manual_document"
        else mutation
    )
    call = getattr(owner, operation)
    call.assert_awaited_once()
    assert call.await_args.kwargs["user_role"] is UserRole.ADMIN


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id,expected", [(1, 200), (2, 409)])
async def test_chunk_edit_keeps_document_permissions_through_composed_service(asgi_client, user_id, expected):
    uow = _make_uow(
        doc=_make_doc(owner_id=1, visibility=DocumentVisibility.INTERNAL_PRIVATE),
        chunk=_make_chunk(owner_id=1, visibility=DocumentVisibility.INTERNAL_PRIVATE.value),
    )
    service = _make_service(uow)
    app.state.container = SimpleNamespace(application=SimpleNamespace(chunk_mutation_service=service))
    sign_in(UserRole.CURATOR.value, user_id)
    response = await asgi_client.put("/documents/1/chunks/10", json={"content": "updated " * 30})
    assert response.status_code == expected

    if expected == 200:
        uow.chunks.update_content.assert_awaited_once()
        uow.vector_outbox.enqueue.assert_awaited_once()
    else:
        uow.chunks.update_content.assert_not_awaited()
        uow.vector_outbox.enqueue.assert_not_awaited()
