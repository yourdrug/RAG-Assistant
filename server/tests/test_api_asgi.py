"""Real application ASGI contracts, keeping auth rules, middleware and handlers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from application.chat import ConversationService, MessageFeedbackService
from application.chunks import AddChunkResult, EditChunkResult
from application.documents import DocumentDeletionJobs, DocumentDTO, DocumentQueryService
from application.spaces import KnowledgeSpaceService
from domain.benchmark.question import DEFAULT_DATASET
from domain.chat.message import Message, MessageRole
from domain.identity.roles import UserRole
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.knowledge.chunk import ChunkRecord
from domain.knowledge.document import Document
from domain.knowledge.enums import DocDomain, DocumentVisibility, SourceType
from domain.knowledge.space import KnowledgeSpace
from domain.runtime.jobs import BackgroundJob, BackgroundJobStatus, JobType
from domain.shared.exceptions import (
    AuthenticationError,
    LLMUnavailableError,
    PermissionDeniedError,
    RateLimitExceededError,
    ValidationError,
)
from domain.shared.pagination import CursorPage
from main import app
from presentation.api import dependencies as deps
from presentation.api.auth_dependencies import get_current_user
from presentation.api.schemas import CurrentUser
from tests.fakes import (
    DEFAULT_SPACE_ID,
    FakePackCatalog,
    FakeUnitOfWork,
    FakeUnitOfWorkFactory,
    require_present,
)
from tests.test_chunk_service import _make_chunk, _make_doc, _make_service, _make_uow


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


def mock_dependency() -> MagicMock:
    """Expose a zero-argument signature for FastAPI dependency inspection."""
    return MagicMock()


@pytest_asyncio.fixture
async def asgi_client() -> AsyncIterator[httpx.AsyncClient]:
    old_overrides = dict(app.dependency_overrides)
    old_container = getattr(app.state, "container", None)
    app.state.container = None
    app.dependency_overrides[deps.create_rate_limiter] = lambda: None
    app.dependency_overrides[deps.create_action_logger] = mock_dependency
    app.dependency_overrides[deps.create_auth_service] = mock_dependency
    app.dependency_overrides[deps.create_api_key_service] = mock_dependency
    app.dependency_overrides[deps.create_api_key_provider] = mock_dependency
    # Neutral pack: only the ``general`` document profile is known.
    app.dependency_overrides[deps.create_domain_registry] = lambda: None
    app.dependency_overrides[deps.create_knowledge_space_service] = neutral_space_service
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]  # Starlette and HTTPX disagree on the ASGI send annotation.
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(old_overrides)
        app.state.container = old_container


class KeyedProfile(GeneralDomainProfile):
    """Neutral document profile registered under another key."""

    def __init__(self, key: str) -> None:
        self.key = key


def law_registry() -> DomainProfileRegistry:
    """Stand-in for the law pack registry: only its profile keys matter to the routes."""
    registry = DomainProfileRegistry()

    for key in ("general", "legal", "decree", "contract"):
        registry.register(KeyedProfile(key))

    return registry


def space_service(registry: DomainProfileRegistry | None = None) -> KnowledgeSpaceService:
    """The only space of the installation; on the stand-in law pack when a registry is given."""
    uow = FakeUnitOfWork()

    if registry is None:
        return KnowledgeSpaceService(FakeUnitOfWorkFactory(uow), FakePackCatalog())

    uow.fake_knowledge_spaces._spaces[DEFAULT_SPACE_ID] = KnowledgeSpace(
        id=DEFAULT_SPACE_ID, name="Право", pack_key="law"
    )
    return KnowledgeSpaceService(FakeUnitOfWorkFactory(uow), FakePackCatalog({"law": registry}))


def neutral_space_service() -> KnowledgeSpaceService:
    """Zero-argument dependency override: FastAPI would read parameters as query fields."""
    return space_service()


def sign_in(role: str = "user", user_id: int = 1) -> None:
    user = CurrentUser(id=user_id, email="asgi@example.org", role=role, kind="internal", is_active=True)
    app.dependency_overrides[get_current_user] = lambda: user


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "law_pack,keys", [(False, ["general"]), (True, ["general", "legal", "decree", "contract"])]
)
async def test_document_domains_list_profiles_of_the_space_pack(
    asgi_client: httpx.AsyncClient, law_pack: bool, keys: list[str]
) -> None:
    sign_in()
    registry = law_registry() if law_pack else None
    app.dependency_overrides[deps.create_knowledge_space_service] = lambda: space_service(registry)
    response = await asgi_client.get("/documents/domains")
    assert response.status_code == 200
    assert [option["key"] for option in response.json()] == keys
    assert all(option["display_name"] for option in response.json())


@pytest.mark.asyncio
async def test_document_domains_require_authentication(asgi_client: httpx.AsyncClient) -> None:
    response = await asgi_client.get("/documents/domains")
    assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/ingest", "/ingest/file"])
@pytest.mark.parametrize("domain", ["legal_typo", "", "AUTO"])
async def test_ingestion_rejects_unknown_domain_before_creating_job(
    asgi_client: httpx.AsyncClient, path: str, domain: str
) -> None:
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
async def test_ingestion_accepts_supported_and_automatic_domains(
    asgi_client: httpx.AsyncClient, path: str, domain: DocDomain
) -> None:
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
@pytest.mark.parametrize("path", ["/ingest", "/ingest/file"])
@pytest.mark.parametrize(
    "law_pack, domain, status",
    [(False, "legal", 422), (True, "legal", 200), (True, "decree", 200), (True, "contract", 200)],
)
async def test_ingestion_domains_come_from_the_space_pack(
    asgi_client: httpx.AsyncClient, path: str, law_pack: bool, domain: str, status: int
) -> None:
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
    registry = law_registry() if law_pack else None
    app.dependency_overrides[deps.create_knowledge_space_service] = lambda: space_service(registry)
    response = await asgi_client.post(path, params={"domain": domain, "file_path": "docs/test.pdf"})
    assert response.status_code == status


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", ["legal_typo", "auto"])
async def test_preview_index_rejects_unknown_domain_before_loading_file(
    asgi_client: httpx.AsyncClient, domain: str
) -> None:
    sign_in(UserRole.ADMIN.value)
    cache = SimpleNamespace(get_bytes=AsyncMock(return_value=None))
    app.dependency_overrides[deps.create_preview_cache] = lambda: cache
    app.dependency_overrides[deps.create_document_command_service] = mock_dependency
    app.dependency_overrides[deps.create_job_service] = mock_dependency
    app.dependency_overrides[deps.create_job_enqueuer] = mock_dependency
    response = await asgi_client.post("/admin/documents/preview/test/index", data={"doc_domain": domain})
    assert response.status_code == 422
    assert "doc_domain" in response.json()["errors"]
    cache.get_bytes.assert_not_awaited()


@pytest.mark.asyncio
async def test_exact_search_reports_full_total_and_requested_page(asgi_client: httpx.AsyncClient) -> None:
    sign_in()
    row = ChunkRecord(3, 10, "test.txt", "matching needle text", 3)
    service = SimpleNamespace(exact_search=AsyncMock(return_value=([row], 53)))
    app.dependency_overrides[deps.create_search_service] = lambda: service
    response = await asgi_client.get(
        "/search", params={"query": "needle", "mode": "icontains", "limit": 2, "offset": 2}
    )
    assert response.status_code == 200
    assert response.json()["total"] == 53
    assert response.json()["limit"] == 2
    assert response.json()["offset"] == 2
    assert [row["chunk_id"] for row in response.json()["results"]] == [3]
    assert service.exact_search.call_args.kwargs["offset"] == 2
    assert service.exact_search.call_args.kwargs["mode"] == "icontains"


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [{"mode": "provision"}, {"unknown": "1"}])
async def test_search_rejects_internal_mode_and_unknown_params(
    asgi_client: httpx.AsyncClient, params: dict[str, str]
) -> None:
    sign_in()
    service = SimpleNamespace(exact_search=AsyncMock())
    app.dependency_overrides[deps.create_search_service] = lambda: service
    response = await asgi_client.get("/search", params={"query": "needle", **params})
    assert response.status_code == 422
    service.exact_search.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("offset", [-1, 0.5, "bad"])
async def test_exact_search_rejects_invalid_offset(asgi_client: httpx.AsyncClient, offset: int) -> None:
    sign_in()
    service = SimpleNamespace(exact_search=AsyncMock())
    app.dependency_overrides[deps.create_search_service] = lambda: service
    response = await asgi_client.get("/search", params={"query": "needle", "offset": offset})
    assert response.status_code == 422
    service.exact_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_benchmark_launch_passes_llm_replay_mode_to_the_job(
    asgi_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    sign_in(role=UserRole.ADMIN.value)
    enqueuer = SimpleNamespace(enqueue_benchmark=AsyncMock())
    jobs = SimpleNamespace(create_job=AsyncMock(return_value=8))
    app.dependency_overrides[deps.create_job_enqueuer] = lambda: enqueuer
    app.dependency_overrides[deps.create_job_service] = lambda: jobs
    app.dependency_overrides[deps.create_benchmark_config] = lambda: SimpleNamespace(
        data_dir=str(tmp_path), retriever_top_k=7, llm_model="judge"
    )
    app.dependency_overrides[deps.create_idempotency_store] = lambda: None
    response = await asgi_client.post("/benchmark", json={"llm_replay_mode": "replay_only"})
    assert response.status_code == 200
    assert enqueuer.enqueue_benchmark.call_args.kwargs["llm_replay_mode"] == "replay_only"
    assert jobs.create_job.call_args.kwargs["dispatch_payload"]["llm_replay_mode"] == "replay_only"
    rejected = await asgi_client.post("/benchmark", json={"llm_replay_mode": "always"})
    assert rejected.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("dataset", [None, "custom"])
async def test_benchmark_launch_enqueues_database_dataset(
    asgi_client: httpx.AsyncClient, tmp_path: Path, dataset: str | None
) -> None:

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
        dataset=dataset or DEFAULT_DATASET,
        top_k=7,
        judge_model="judge",
        job_id=8,
        space_id=DEFAULT_SPACE_ID,
        llm_replay_mode="off",
    )
    assert list(tmp_path.iterdir()) == []  # noqa: ASYNC240 - Small local fixture I/O in the test setup.


@pytest.mark.asyncio
async def test_real_auth_required(asgi_client: httpx.AsyncClient) -> None:
    response = await asgi_client.get("/conversations")
    assert response.status_code == 401
    assert "message" in response.json()
    assert response.headers["X-Request-ID"]


@pytest.mark.asyncio
@pytest.mark.parametrize("role,status", [("user", 403), ("curator", 403), ("admin", 200)])
async def test_benchmark_compare_route_and_roles(
    asgi_client: httpx.AsyncClient, role: str, status: int
) -> None:
    sign_in(role)
    service = SimpleNamespace(compare=AsyncMock(return_value=([], {})))
    app.dependency_overrides[deps.create_benchmark_run_service] = lambda: service
    response = await asgi_client.get("/admin/benchmark/runs/compare?ids=1,2")
    assert response.status_code == status

    if status == 200:
        assert response.json() == {"runs": [], "diff": {}, "diagnosis_transitions": []}
        service.compare.assert_awaited_once_with([1, 2])
    else:
        service.compare.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename,data",
    [
        ("bad.exe", b"text"),
        ("bad.pdf", b"%PDF-broken"),
        ("legacy.doc", b"\xd0\xcf\x11\xe0 ole2"),
        ("legacy.DOC", b"\xd0\xcf\x11\xe0 ole2"),
    ],
)
async def test_file_validation_returns_422_before_upload(
    asgi_client: httpx.AsyncClient, filename: str, data: bytes
) -> None:
    sign_in("admin")
    command = SimpleNamespace(upload=AsyncMock())
    app.dependency_overrides[deps.create_document_command_service] = lambda: command
    app.dependency_overrides[deps.create_job_service] = mock_dependency
    app.dependency_overrides[deps.create_job_enqueuer] = mock_dependency
    app.dependency_overrides[deps.create_upload_config] = lambda: SimpleNamespace(max_upload_size_mb=1)
    app.dependency_overrides[deps.create_idempotency_store] = mock_dependency
    response = await asgi_client.post(
        "/documents", files={"file": (filename, data)}, data={"visibility": "internal_public"}
    )
    assert response.status_code == 422
    assert "file" in response.json()["errors"]
    command.upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_conversation_cursor_and_ownership(asgi_client: httpx.AsyncClient) -> None:
    factory = FakeUnitOfWorkFactory()
    factory._uow.fake_conversations._convs[7] = {"id": 7, "user_id": 1, "summary": None}
    factory._uow.fake_messages._messages = [
        Message(id=i, conversation_id=7, role=MessageRole.USER, content=str(i)) for i in range(1, 106)
    ]
    service = ConversationService(factory, MagicMock(), MagicMock())
    app.dependency_overrides[deps.create_conversation_service] = lambda: service
    feedback = MessageFeedbackService(factory, MagicMock())
    app.dependency_overrides[deps.create_message_feedback_service] = lambda: feedback
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
async def test_message_feedback_routes_map_ownership_and_invariants(asgi_client: httpx.AsyncClient) -> None:
    factory = FakeUnitOfWorkFactory()
    factory._uow.fake_conversations._convs[7] = {"id": 7, "user_id": 1, "summary": None}
    factory._uow.fake_messages._messages = [
        Message(id=1, conversation_id=7, role=MessageRole.USER, content="q"),
        Message(id=2, conversation_id=7, role=MessageRole.ASSISTANT, content="a"),
    ]
    feedback = MessageFeedbackService(factory, MagicMock(redact=lambda text: text))
    app.dependency_overrides[deps.create_message_feedback_service] = lambda: feedback
    sign_in(user_id=1)

    rated = await asgi_client.put(
        "/chat/messages/2/feedback", json={"rating": "down", "reasons": ["outdated"], "comment": "old"}
    )
    assert rated.status_code == 200
    assert rated.json()["reasons"] == ["outdated"]
    invalid_reason = await asgi_client.put(
        "/chat/messages/2/feedback", json={"rating": "down", "reasons": ["invalid"]}
    )
    assert invalid_reason.status_code == 422
    assert "reasons.0" in invalid_reason.json()["errors"]

    invalid = await asgi_client.put(
        "/chat/messages/2/feedback", json={"rating": "up", "reasons": ["outdated"]}
    )
    assert invalid.status_code == 422
    assert (await asgi_client.put("/chat/messages/1/feedback", json={"rating": "up"})).status_code == 404
    assert (await asgi_client.delete("/chat/messages/2/feedback")).status_code == 204

    sign_in(user_id=2)
    assert (await asgi_client.put("/chat/messages/2/feedback", json={"rating": "up"})).status_code == 404
    assert factory._uow.fake_message_feedback.items == {}


@pytest.mark.asyncio
async def test_missing_route_uses_public_error_envelope(
    asgi_client: httpx.AsyncClient,
) -> None:
    missing = await asgi_client.get("/nonexistent-path")
    assert missing.status_code == 404
    assert missing.json() == {"message": "Not Found"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,code,message,retry_after",
    [
        (
            ValidationError("Choose a knowledge space", field="space_id"),
            "validation_error",
            "Choose a knowledge space",
            None,
        ),
        (PermissionDeniedError(), "forbidden", "Недостаточно прав", None),
        (AuthenticationError(), "unauthorized", "Не авторизован", None),
        (RateLimitExceededError(12, 1), "rate_limit_exceeded", "Too many requests", 12),
        (LLMUnavailableError(), "llm_unavailable", "LLM временно недоступен", 30),
        (RuntimeError("private diagnostics"), "internal_error", "Внутренняя ошибка сервера", None),
    ],
)
async def test_chat_stream_returns_actionable_client_errors(
    asgi_client: httpx.AsyncClient, error: Exception, code: str, message: str, retry_after: int | None
) -> None:
    sign_in()
    service = MagicMock()
    service.stream_chat.side_effect = error
    app.dependency_overrides[deps.create_chat_service] = lambda: service
    response = await asgi_client.post("/chat", json={"question": "test"})
    assert response.status_code == 200
    assert "event: error" in response.text
    assert f'"code": "{code}"' in response.text
    assert message in response.text
    assert "event: done" not in response.text
    assert "private diagnostics" not in response.text

    if retry_after is not None:
        assert f'"retry_after": {retry_after}' in response.text


@pytest.mark.asyncio
async def test_unexpected_error_preserves_cors_and_request_id(asgi_client: httpx.AsyncClient) -> None:
    sign_in()
    service = SimpleNamespace(sync_chat=AsyncMock(side_effect=RuntimeError("private diagnostics")))
    app.dependency_overrides[deps.create_chat_service] = lambda: service
    origin = "https://ui.example.org"
    response = await asgi_client.post("/chat/sync", json={"question": "test"}, headers={"Origin": origin})

    assert response.status_code == 500
    assert response.json() == {"message": "Внутренняя ошибка сервера"}
    assert response.headers["access-control-allow-origin"] in (origin, "*")
    assert "retry-after" in response.headers["access-control-expose-headers"].lower()
    assert response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_chat_outage_preserves_retry_after(asgi_client: httpx.AsyncClient) -> None:

    sign_in()
    service = SimpleNamespace(sync_chat=AsyncMock(side_effect=LLMUnavailableError()))
    app.dependency_overrides[deps.create_chat_service] = lambda: service
    response = await asgi_client.post("/chat/sync", json={"question": "test"})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"
    assert "message" in response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id,role,status", [(1, "user", 200), (2, "user", 409), (2, "admin", 200)])
async def test_private_document_ownership_through_real_route(
    asgi_client: httpx.AsyncClient, user_id: int, role: str, status: int
) -> None:

    factory = FakeUnitOfWorkFactory()
    document = await factory._uow.fake_documents.save(
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
async def test_document_delete_queues_only_after_authorization(
    asgi_client: httpx.AsyncClient,
    user_id: int,
    role: UserRole,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    factory = FakeUnitOfWorkFactory()
    document = await factory._uow.fake_documents.save(
        Document(
            filename="private.pdf",
            owner_id=1,
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
        )
    )
    jobs = factory._uow.fake_background_jobs
    monkeypatch.setattr(
        jobs,
        "create",
        AsyncMock(
            return_value=BackgroundJob(id=42, related_id=document.id, job_type=JobType.DOCUMENT_DELETION)
        ),
    )
    enqueuer = SimpleNamespace(enqueue_document_deletion=AsyncMock())
    service = DocumentDeletionJobs(factory, enqueuer)
    app.dependency_overrides[deps.create_document_deletion_jobs] = lambda: service
    sign_in(role.value, user_id)
    response = await asgi_client.delete(f"/documents/{document.id}")
    assert response.status_code == expected
    assert await factory._uow.fake_documents.get_by_id(require_present(document.id)) == document

    if expected == 202:
        assert response.json() == {
            "status": BackgroundJobStatus.PENDING.value,
            "document_id": document.id,
            "job_id": 42,
        }
        enqueuer.enqueue_document_deletion.assert_awaited_once_with(
            document_id=document.id, user_id=user_id, job_id=42
        )
        monkeypatch.setattr(
            jobs,
            "get_by_id",
            AsyncMock(
                return_value=BackgroundJob(id=42, related_id=document.id, job_type=JobType.DOCUMENT_DELETION)
            ),
        )
        progress = await asgi_client.get(f"/documents/{document.id}/deletion/42")
        assert progress.status_code == 200
        assert progress.json()["status"] == BackgroundJobStatus.PENDING.value
        await factory._uow.fake_documents.delete(require_present(document.id))
        assert (await asgi_client.get(f"/documents/{document.id}/deletion/42")).status_code == 404
    else:
        cast("AsyncMock", jobs.create).assert_not_awaited()
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
async def test_chunk_routes_resolve_composed_services(
    asgi_client: httpx.AsyncClient, method: str, path: str, payload: dict[str, Any] | None, operation: str
) -> None:
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
async def test_chunk_edit_keeps_document_permissions_through_composed_service(
    asgi_client: httpx.AsyncClient, user_id: int, expected: int
) -> None:
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


@pytest.mark.asyncio
async def test_multipart_upload_preserves_metadata_and_job_principal(asgi_client: httpx.AsyncClient) -> None:
    sign_in(UserRole.ADMIN.value, user_id=9)
    result = SimpleNamespace(
        id=41, storage_key="docs/regulation.txt", filename="regulation.txt", owner_id=21, replace_id=7
    )
    commands = SimpleNamespace(upload=AsyncMock(return_value=result))
    jobs = SimpleNamespace(create_job=AsyncMock(return_value=8))
    enqueuer = SimpleNamespace(enqueue_document_processing=AsyncMock())
    app.dependency_overrides[deps.create_document_command_service] = lambda: commands
    app.dependency_overrides[deps.create_job_service] = lambda: jobs
    app.dependency_overrides[deps.create_job_enqueuer] = lambda: enqueuer
    app.dependency_overrides[deps.create_upload_config] = lambda: SimpleNamespace(max_upload_size_mb=1)
    app.dependency_overrides[deps.create_idempotency_store] = lambda: None
    registry = law_registry()
    app.dependency_overrides[deps.create_domain_registry] = lambda: registry

    response = await asgi_client.post(
        "/documents",
        files={"file": ("regulation.txt", b"legal text", "text/plain")},
        data={
            "visibility": DocumentVisibility.CLIENT_PRIVATE.value,
            "group_id": "5",
            "client_id": "21",
            "rename_on_conflict": "true",
            "doc_domain": "legal",
            "replaces_document_id": "7",
            "space_id": "3",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"status": "processing", "document_id": 41, "filename": "regulation.txt"}
    commands.upload.assert_awaited_once_with(
        filename="regulation.txt",
        file_data=b"legal text",
        visibility=DocumentVisibility.CLIENT_PRIVATE.value,
        group_id=5,
        client_id=21,
        user_id=9,
        user_kind="internal",
        user_role=UserRole.ADMIN.value,
        rename_on_conflict=True,
        doc_domain="legal",
        replaces_document_id=7,
        space_id=3,
    )
    # Upload owns the durable job; the HTTP path never publishes to Redis.
    jobs.create_job.assert_not_awaited()
    enqueuer.enqueue_document_processing.assert_not_awaited()


@pytest.mark.asyncio
async def test_question_filters_reach_service_and_response_preserves_total(
    asgi_client: httpx.AsyncClient,
) -> None:
    sign_in(UserRole.ADMIN.value)
    questions = SimpleNamespace(list=AsyncMock(return_value=([], 53)))
    app.dependency_overrides[deps.create_benchmark_question_service] = lambda: questions
    response = await asgi_client.get(
        "/admin/benchmark/questions",
        params={
            "dataset": "sample",
            "tag": "law",
            "search": "text",
            "is_active": "false",
            "limit": 4,
            "offset": 8,
        },
    )
    assert response.status_code == 200
    assert response.json() == {"questions": [], "total": 53}
    questions.list.assert_awaited_once_with(
        dataset="sample", tag="law", search="text", is_active=False, limit=4, offset=8
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role,expected", [("admin", 204), ("curator", 403), ("user", 403)])
async def test_only_admin_can_assign_client_space(
    asgi_client: httpx.AsyncClient, role: str, expected: int
) -> None:
    sign_in(role)
    uow = FakeUnitOfWork()
    uow.fake_users.add_user(9, kind="client")
    service = KnowledgeSpaceService(FakeUnitOfWorkFactory(uow), FakePackCatalog())
    app.dependency_overrides[deps.create_knowledge_space_service] = lambda: service
    response = await asgi_client.put("/spaces/clients/9", json={"space_ids": [DEFAULT_SPACE_ID]})
    assert response.status_code == expected
    client = await uow.users.get_by_id(9)
    assert client is not None
    assert client.client_space_ids == (frozenset({DEFAULT_SPACE_ID}) if expected == 204 else frozenset())


@pytest.mark.asyncio
async def test_client_space_assignment_validation_and_listing(asgi_client: httpx.AsyncClient) -> None:
    sign_in("admin")
    uow = FakeUnitOfWork()
    uow.fake_users.add_user(9, kind="client")
    uow.fake_users.add_user(10, kind="internal")
    service = KnowledgeSpaceService(FakeUnitOfWorkFactory(uow), FakePackCatalog())
    app.dependency_overrides[deps.create_knowledge_space_service] = lambda: service
    assert (await asgi_client.put("/spaces/clients/9", json={"space_ids": [999]})).status_code == 404
    assert (
        await asgi_client.put("/spaces/clients/999", json={"space_ids": [DEFAULT_SPACE_ID]})
    ).status_code == 404
    assert (
        await asgi_client.put("/spaces/clients/10", json={"space_ids": [DEFAULT_SPACE_ID]})
    ).status_code == 409
    assert (await asgi_client.put("/spaces/clients/9", json={"space_ids": [-1]})).status_code == 422
    assert (await asgi_client.put("/spaces/clients/9", json={})).status_code == 422
    assert (
        await asgi_client.put("/spaces/clients/9", json={"space_ids": [DEFAULT_SPACE_ID]})
    ).status_code == 204

    current = CurrentUser(
        id=9, email="client@example.org", role="user", kind="client", is_active=True, api_key_id=7
    )
    app.dependency_overrides[get_current_user] = lambda: current
    listed = await asgi_client.get("/spaces")
    assert listed.status_code == 200
    assert [space["id"] for space in listed.json()] == [DEFAULT_SPACE_ID]
    assert listed.json()[0]["can_upload"] is False
    assert (await asgi_client.put("/spaces/clients/9", json={"space_ids": []})).status_code == 403

    sign_in("admin")
    assert (await asgi_client.put("/spaces/clients/9", json={"space_ids": []})).status_code == 204
    app.dependency_overrides[get_current_user] = lambda: current
    assert (await asgi_client.get("/spaces")).json() == []
