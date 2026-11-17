"""Production trace capture, cache parity and viewer-scoped fragment reads."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

from fastapi import HTTPException
from langchain_core.documents import Document as LCDocument
import pytest

from application.chat.traces import ChatTraceService, public_trace
from application.rag.dto import PreparedQuery, TerminalResponse
from application.rag.stream_events import PipelineMetaEvent, TextChunk
from composition.service_providers import create_rag_service
from config import settings
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.chunk import ChunkRecord
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentVisibility
from domain.rag.breadth import Breadth
from domain.rag.context import RagContext
from domain.rag.outcome import AnswerOutcome
from domain.shared.exceptions import EntityNotFoundError
from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile, evidence_profile_scope
from infrastructure.ml.rag import rag_steps
from infrastructure.ml.rag.benchmark_evidence import (
    BenchmarkEvidence,
    active_evidence,
    authorize_capture,
    begin_attempt,
    capture_completion,
    capture_gate,
    capture_prompt,
    capture_prompt_delivery,
    capture_stage,
)
from infrastructure.ml.rag.chat_trace import (
    MAX_TRACE_BYTES,
    bounded_trace,
    chat_trace_scope,
    refusal_stage,
    stored_size,
)
from presentation.api.routes.admin_chat_logs import TraceChunkRequest, get_trace, get_trace_chunks
from presentation.api.schemas.auth import CurrentUser
from tests.rag_stage_fixtures import prepared_query, stage_fixture
from tests.test_chat_contracts import chat_fixture
from tests.test_chat_logs import attempt_fixture, run_attempt
from tests.test_rag_pipeline import make_rag


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

    from application.rag.dto import RagRequest
    from application.rag.stream_events import StreamEvent
    from domain.chat.chat_log import ChatLog


@pytest.fixture(autouse=True)
def neutral_profile() -> Iterator[None]:
    with evidence_profile_scope(GenericEvidenceProfile()):
        yield


CONTENT = "PRIVATE document text"


def capture_example() -> None:
    doc = LCDocument(page_content=CONTENT, metadata={"chunk_id": 7, "document_id": 9, "heading": CONTENT})
    begin_attempt("Q", "Q")
    authorize_capture([doc])
    capture_stage("candidates", [doc])
    capture_stage("reranked", [(doc, 0.9)], score_kind="reranker")
    capture_prompt([doc], [doc], CONTENT, messages=[{"role": "system", "content": "SECRET PROMPT"}])
    capture_gate("relevance", "Q", CONTENT, {"is_relevant": False, "reason": CONTENT * 100})
    capture_prompt_delivery(sent=False)
    capture_completion(AnswerOutcome.NOT_FOUND)


def test_production_projection_keeps_ids_but_no_document_or_prompt_text() -> None:
    with chat_trace_scope(None):
        evidence = active_evidence.get()
        assert evidence is not None
        capture_example()
        assert CONTENT not in str(evidence.attempts[0]["stages"])
        assert "SECRET PROMPT" not in str(asdict(evidence))
    data = evidence.trace_data
    assert data["schema_version"] == 1
    assert data["attempts"][0]["stages"]["prompt"]["documents"][0]["metadata"] == {
        "chunk_id": 7,
        "document_id": 9,
    }
    assert len(data["attempts"][0]["checks"][0]["result"]["reasoning"]) == 500
    assert CONTENT not in str(public_trace(data))
    assert active_evidence.get() is None


def test_benchmark_capture_stays_identical_inside_production_scope() -> None:
    def run(wrapped: bool) -> dict[str, Any]:
        evidence = BenchmarkEvidence()
        token = active_evidence.set(evidence)
        try:
            if wrapped:
                with chat_trace_scope(None):
                    capture_example()
            else:
                capture_example()

            return asdict(evidence)
        finally:
            active_evidence.reset(token)

    assert run(False) == run(True)
    assert CONTENT in str(run(True))


@pytest.mark.parametrize("outcome", list(AnswerOutcome))
def test_trace_all_outcomes_except_input_blocked(outcome: AnswerOutcome) -> None:
    with chat_trace_scope(None):
        evidence = active_evidence.get()
        assert evidence is not None
        evidence.trace_data["outcome"] = outcome
        capture_completion(outcome)
    assert bool(evidence.trace_data) == (outcome != AnswerOutcome.INPUT_BLOCKED)


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError, GeneratorExit])
def test_capture_survives_failure_and_generator_close(error: type[BaseException]) -> None:
    with pytest.raises(error), chat_trace_scope(None):  # noqa: PT012 - verify cleanup around the failing scope
        evidence = active_evidence.get()
        assert evidence is not None
        capture_example()
        raise error
    assert evidence is not None
    assert evidence.trace_data["completion"] == (
        AnswerOutcome.ERROR if error is RuntimeError else AnswerOutcome.CANCELLED
    )
    assert active_evidence.get() is None


def test_disabled_and_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "trace_enabled", False)
    with chat_trace_scope(None):
        assert active_evidence.get() is None
    data = {
        "schema_version": 1,
        "attempts": [
            {
                "exclusions": [],
                "stages": {
                    "candidates": {
                        "documents": [
                            {"metadata": {"chunk_id": i}, "content_hash": "x" * 1000} for i in range(200)
                        ]
                    }
                },
            }
        ],
    }
    result = bounded_trace(data)
    assert result["truncated"]
    assert len(json.dumps(result, ensure_ascii=False).encode()) <= MAX_TRACE_BYTES


def test_size_limit_counts_numbers_as_postgres_prints_them() -> None:
    # Python writes 1e-07 (5 bytes); jsonb::text writes 0.0000001 (9 bytes).
    documents = [{"metadata": {"chunk_id": i}, "score": 1e-07, "rrf_score": 1e-07} for i in range(3000)]
    data = {
        "schema_version": 1,
        "attempts": [{"exclusions": [], "stages": {"retrieved": {"documents": documents}}}],
    }
    jsonb_text = json.dumps(data, ensure_ascii=False).replace("1e-07", "0.0000001")
    assert stored_size(data) >= len(jsonb_text.encode())

    result = bounded_trace(data)
    assert result["truncated"]
    jsonb_text = json.dumps(result, ensure_ascii=False).replace("1e-07", "0.0000001")
    assert len(jsonb_text.encode()) <= MAX_TRACE_BYTES


@pytest.mark.parametrize(
    "checks, expected",
    [
        ([], "retrieval"),
        ([{"stage": "relevance", "result": {"is_relevant": True}}], "retrieval"),
        ([{"stage": "relevance", "result": {"is_relevant": False}}], "relevance"),
        (
            [
                {"stage": "relevance", "result": {"is_relevant": False}},
                {"stage": "sufficiency", "result": {"is_sufficient": False}},
            ],
            "sufficiency",
        ),
        (
            [
                {"stage": "relevance", "result": {"is_relevant": False}},
                {"stage": "sufficiency", "result": {"is_sufficient": True}},
            ],
            "relevance",
        ),
    ],
)
def test_refusal_stage_is_last_negative_gate(checks: list[dict[str, Any]], expected: str) -> None:
    assert refusal_stage([{"checks": checks}]) == expected


@pytest.mark.asyncio
async def test_production_capture_keeps_cache_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    rag = make_rag()
    state = stage_fixture(
        question="Q",
        ctx=RagContext(user=UserContext(1, UserKind.INTERNAL)),
        rag=replace(rag, features=replace(rag.features, cache_enabled=True)),
    )
    query = prepared_query(state)
    revision = AsyncMock(return_value=None)
    monkeypatch.setattr(rag_steps, "get_corpus_revision", revision)
    operations = rag_steps.RagOperations(SimpleNamespace(), document_access=SimpleNamespace())

    for capture in (False, True):
        token = active_evidence.set(BenchmarkEvidence(benchmark_mode=False) if capture else None)
        try:
            assert await operations.lookup(query) is query
        finally:
            active_evidence.reset(token)

    assert revision.await_count == 2


def trace_fixture() -> SimpleNamespace:
    f = chat_fixture()
    with chat_trace_scope(None):
        evidence = active_evidence.get()
        assert evidence is not None
        capture_example()
    f.uow.chat_logs.get_by_id.side_effect = lambda _id, ids: (
        SimpleNamespace(id=1) if ids is None or 10 in ids else None
    )
    f.uow.chat_logs.get_trace.return_value = evidence.trace_data
    f.uow.documents.get_by_id.return_value = Document(
        id=9, visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=20
    )
    f.uow.chunks.get_by_id.return_value = ChunkRecord(
        7, 9, "file", CONTENT, 0, visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=20
    )
    f.traces = ChatTraceService(f.factory)
    return f


@pytest.mark.asyncio
async def test_admin_listing_reads_client_private_and_detects_changed_or_missing() -> None:
    f = trace_fixture()
    viewer = UserContext(1, UserKind.INTERNAL, UserRole.ADMIN)
    result = await f.traces.read_chunks(1, [7], viewer, None)
    assert result["chunks"] == [{"chunk_id": 7, "content": CONTENT, "changed": False}]
    assert result["checks"][0]["result"]["reasoning"]
    f.uow.chunks.get_by_id.return_value.content = "new"
    assert (await f.traces.read_chunks(1, [7], viewer, None))["chunks"][0]["changed"]
    assert not (await f.traces.read_chunks(1, [7], viewer, None))["checks"]
    f.uow.chunks.get_by_id.return_value = None
    assert (await f.traces.read_chunks(1, [7], viewer, None))["chunks"] == [{"chunk_id": 7, "missing": True}]


@pytest.mark.asyncio
async def test_curator_log_scope_does_not_grant_fragment_access() -> None:
    f = trace_fixture()
    viewer = UserContext(1, UserKind.INTERNAL, UserRole.CURATOR)
    assert "reasoning" not in str(await f.traces.get_trace(1, (1, 10)))
    result = await f.traces.read_chunks(1, [7], viewer, (1, 10))
    assert result == {"chunks": [{"chunk_id": 7, "hidden": True}], "checks": []}
    f.uow.chunks.get_by_id.assert_not_awaited()
    with pytest.raises(EntityNotFoundError):
        await f.traces.get_trace(1, (1, 11))
    with pytest.raises(EntityNotFoundError):
        await f.traces.read_chunks(1, [7], viewer, (1, 11))
    with pytest.raises(EntityNotFoundError):
        await f.traces.read_chunks(1, [999], viewer, (1, 10))


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [AnswerOutcome.GENERATED, AnswerOutcome.ERROR, AnswerOutcome.CANCELLED])
async def test_trace_is_written_in_attempt_transaction(outcome: AnswerOutcome) -> None:
    f = attempt_fixture(outcome)
    original = f.rag.stream.side_effect
    data = {"schema_version": 1, "attempts": []}

    async def stream(**kwargs: Any) -> AsyncGenerator[StreamEvent]:
        async for event in original(**kwargs):
            yield replace(event, trace_data=data) if isinstance(event, PipelineMetaEvent) else event

    async def save(entry: ChatLog) -> None:
        entry.id = 42

    f.uow.chat_logs.save.side_effect = save
    f.rag.stream.side_effect = stream
    with suppress(RuntimeError, asyncio.CancelledError):
        await run_attempt(f, "stream")
    f.uow.chat_logs.save_trace.assert_awaited_once_with(42, data)


@pytest.mark.asyncio
async def test_routes_use_curator_scope_and_unknown_chunk_is_404() -> None:
    f = trace_fixture()
    f.service.traces = f.traces
    user = CurrentUser(id=1, email="viewer@example.test", kind="internal", role="curator", is_active=True)
    assignments = Mock(
        get_scope=AsyncMock(return_value={"managed_client_ids": [], "managed_internal_ids": []})
    )
    with pytest.raises(HTTPException) as error:
        await get_trace(1, user, f.service, assignments)
    assert error.value.status_code == 404
    assignments.get_scope.return_value["managed_client_ids"] = [10]
    result = await get_trace(1, user, f.service, assignments)
    assert result["schema_version"] == 1
    with pytest.raises(HTTPException) as error:
        await get_trace_chunks(1, TraceChunkRequest(chunk_ids=[123]), user, f.service, assignments)
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_trace_does_not_change_cached_answer_or_store(monkeypatch: pytest.MonkeyPatch) -> None:
    rag = make_rag()
    state = stage_fixture(
        question="Q",
        ctx=RagContext(user=UserContext(1, UserKind.INTERNAL)),
        rag=replace(rag, features=replace(rag.features, cache_enabled=True)),
    )
    query = prepared_query(state)
    monkeypatch.setattr(rag_steps, "get_corpus_revision", AsyncMock(return_value="revision"))
    monkeypatch.setattr(rag_steps, "check_cache", AsyncMock(return_value={"answer": "cached"}))
    monkeypatch.setattr(rag_steps, "cache_is_accessible", AsyncMock(return_value=True))
    store = AsyncMock()
    monkeypatch.setattr(rag_steps, "store_answer_cache", store)

    async def hit(*_args: Any, **_kwargs: Any) -> AsyncGenerator[StreamEvent]:
        yield TextChunk("cached")

    monkeypatch.setattr(rag_steps, "handle_cache_hit", hit)
    operations = rag_steps.RagOperations(SimpleNamespace(), document_access=SimpleNamespace())
    results = []

    for capture in (False, True):
        token = active_evidence.set(BenchmarkEvidence(benchmark_mode=False) if capture else None)
        try:
            result = await operations.lookup(query)
            assert isinstance(result, TerminalResponse)
            results.append(result)
            retrieval = SimpleNamespace(
                prepared=replace(query, cache_revision="revision"),
                documents=(),
                breadth=Breadth.NARROW,
                domain="general",
            )
            completed: Any = SimpleNamespace(
                answer=SimpleNamespace(context=SimpleNamespace(retrieval=retrieval), text="answer"),
                sources=[],
                confidence=0.8,
            )
            await operations.store(completed)
        finally:
            active_evidence.reset(token)

    assert results[0] == results[1]
    assert store.await_count == 2
    assert store.await_args_list[0] == store.await_args_list[1]


@pytest.mark.asyncio
async def test_real_stream_delivers_completed_trace_to_chat_service(monkeypatch: pytest.MonkeyPatch) -> None:
    service = create_rag_service(ml_clients=Mock(), document_access=Mock())
    operations = service.preparation

    async def condense(request: RagRequest) -> PreparedQuery:
        return PreparedQuery(request, request.question)

    monkeypatch.setattr(operations, "condense", condense)
    monkeypatch.setattr(operations, "scan", AsyncMock(return_value=None))
    monkeypatch.setattr(
        operations,
        "lookup",
        AsyncMock(
            return_value=TerminalResponse(
                (TextChunk("cached"), PipelineMetaEvent(outcome=AnswerOutcome.CACHE_HIT)),
                outcome=AnswerOutcome.CACHE_HIT,
            )
        ),
    )
    events = [
        event async for event in service.stream("Q", [], RagContext(user=UserContext(1, UserKind.INTERNAL)))
    ]
    meta = [event for event in events if isinstance(event, PipelineMetaEvent)][-1]
    assert meta.trace_data is not None
    assert meta.trace_data["completion"] == AnswerOutcome.CACHE_HIT
    assert meta.trace_data["schema_version"] == 1
    assert active_evidence.get() is None


@pytest.mark.asyncio
async def test_closing_real_stream_finalizes_shared_trace() -> None:
    service = create_rag_service(ml_clients=Mock(), document_access=Mock())
    stream = service.stream("Q", [], RagContext(user=UserContext(1, UserKind.INTERNAL)))
    meta = await anext(stream)
    assert isinstance(meta, PipelineMetaEvent)
    await stream.aclose()
    assert meta.trace_data is not None
    assert meta.trace_data["completion"] == AnswerOutcome.CANCELLED
    assert active_evidence.get() is None
