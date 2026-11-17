"""Answer attempts: parity, cancellation, safe snapshots and administrative scope."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from datetime import date
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

from fastapi import HTTPException
from pydantic import SecretStr
import pytest

from application.rag.dto import RagRequest
from application.rag.service import auxiliary_tokens, auxiliary_usage
from application.rag.stream_events import PipelineMetaEvent, SourcesEvent, TextChunk, UsageReport
from config import settings
from domain.chat.chat_log import ChatLog
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.rag.breadth import Breadth
from domain.rag.context import RagContext
from domain.rag.exceptions import ContextBudgetExceededError
from domain.rag.outcome import AnswerOutcome
from infrastructure.ml.cache.keys import config_snapshot
from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile, evidence_profile_scope
from presentation.api.routes.admin_chat_logs import get_chat_log, get_config_snapshot
from tests.test_chat_contracts import chat_fixture
from tests.test_rag_pipeline import make_rag


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from types import SimpleNamespace

    from application.rag.stream_events import StreamEvent
    from domain.chat.message import Message


def attempt_fixture(outcome: AnswerOutcome) -> SimpleNamespace:
    fixture = chat_fixture()
    saved_messages: list[Message] = []

    async def save(message: Message) -> None:
        message.id = len(saved_messages) + 1
        saved_messages.append(message)

    fixture.uow.messages.save.side_effect = save
    fixture.saved_messages = saved_messages
    fixture.snapshot = {"rag": {"fetch_k": 30}, "space": {"id": 1, "pack_key": "generic"}}

    async def stream(**_kwargs: object) -> AsyncGenerator[StreamEvent]:
        yield PipelineMetaEvent(
            outcome=outcome,
            breadth=Breadth.BROAD,
            domain="legal",
            request_id="request-1",
            model_used="actual-model",
            config_hash="a" * 64,
            config_data=fixture.snapshot,
            config_version="version-1",
            evidence_profile="generic@1",
            candidates_count=15,
            top_rerank_score=0.8,
            condensed_query="condensed",
            sub_queries=["first", "second"],
            refined_query="refined",
            aux_tokens=12,
            retrieval_ms=10,
            rerank_ms=20,
            generation_sec=0.3,
        )
        yield TextChunk("partial answer")

        if outcome == AnswerOutcome.ERROR:
            msg = "private@example.com must never be stored as an error"
            raise RuntimeError(msg)

        if outcome == AnswerOutcome.CANCELLED:
            raise asyncio.CancelledError

        yield SourcesEvent([{"document_id": 9}], 0.7, UsageReport(30, 40, "actual-model"))

    fixture.rag.stream.side_effect = stream
    return fixture


async def run_attempt(fixture: SimpleNamespace, mode: str) -> list[Any]:
    if mode == "stream":
        return [
            event
            async for event in fixture.service.stream_chat(
                "question",
                7,
                1,
                UserKind.INTERNAL,
                UserRole.USER,
                depth="narrow",
                as_of_date=date(2026, 1, 2),
            )
        ]

    return [
        await fixture.service.sync_chat(
            "question", 7, 1, UserKind.INTERNAL, UserRole.USER, depth="narrow", as_of_date=date(2026, 1, 2)
        )
    ]


def check_attempt(fixture: SimpleNamespace, outcome: AnswerOutcome, events: list[Any]) -> dict[str, Any]:
    entry = fixture.uow.chat_logs.save.await_args.args[0]
    assert entry.outcome == outcome
    assert entry.space_id == 1
    assert entry.pack_key == "generic"
    assert entry.breadth == "broad"
    assert entry.domain == "legal"
    assert entry.depth_requested == "narrow"
    assert entry.candidates_count == 15
    assert entry.top_rerank_score == 0.8
    fixture.uow.chat_logs.save_snapshot.assert_awaited_once_with("a" * 64, fixture.snapshot)

    if outcome in {AnswerOutcome.ERROR, AnswerOutcome.CANCELLED}:
        assert not fixture.saved_messages
        assert entry.message_id is None
        assert entry.answer == "partial answer"
        assert "private@example.com" not in repr(asdict(entry))
    else:
        assert [message.role for message in fixture.saved_messages] == ["user", "assistant"]
        assert entry.message_id == fixture.saved_messages[-1].id == 2
        assert entry.sources_count == 1
        assert (entry.input_tokens, entry.output_tokens, entry.aux_tokens) == (30, 40, 12)
        assert events[-1].message_id == entry.message_id

    record: dict[str, Any] = asdict(entry)

    for name in ("creation_date", "latency_ms", "ttft_ms"):
        record.pop(name)

    return record


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", list(AnswerOutcome))
async def test_sync_and_stream_record_same_attempt(outcome: AnswerOutcome) -> None:
    records = []

    for mode in ("stream", "sync"):
        fixture = attempt_fixture(outcome)
        events = []

        if outcome == AnswerOutcome.ERROR:
            with pytest.raises(RuntimeError, match="must never be stored"):
                await run_attempt(fixture, mode)
        elif outcome == AnswerOutcome.CANCELLED:
            with pytest.raises(asyncio.CancelledError):
                await run_attempt(fixture, mode)
        else:
            events = await run_attempt(fixture, mode)

        records.append(check_attempt(fixture, outcome, events))

    assert records[0] == records[1]


@pytest.mark.asyncio
async def test_closing_generator_records_cancelled() -> None:
    fixture = attempt_fixture(AnswerOutcome.GENERATED)
    stream = fixture.service.stream_chat("question", 7, 1, UserKind.INTERNAL, UserRole.USER)
    assert await anext(stream) == TextChunk("partial answer")
    await stream.aclose()
    entry = fixture.uow.chat_logs.save.await_args.args[0]
    assert entry.outcome == AnswerOutcome.CANCELLED
    assert entry.message_id is None
    assert not fixture.saved_messages


@pytest.mark.asyncio
async def test_task_cancellation_detaches_log_write() -> None:
    fixture = chat_fixture()
    generating = asyncio.Event()
    writing = asyncio.Event()
    release_write = asyncio.Event()

    async def stream(**_kwargs: object) -> AsyncGenerator[StreamEvent]:
        yield TextChunk("partial")
        generating.set()
        await asyncio.Event().wait()

    async def save(_entry: ChatLog) -> None:
        writing.set()
        await release_write.wait()

    fixture.rag.stream.side_effect = stream
    fixture.uow.chat_logs.save.side_effect = save

    async def consume() -> None:
        async for _event in fixture.service.stream_chat("question", 7, 1, UserKind.INTERNAL, UserRole.USER):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(generating.wait(), 1)
    task.cancel()
    await asyncio.wait_for(writing.wait(), 1)
    task.cancel()  # A second cancellation must not cancel the detached transaction.
    with pytest.raises(asyncio.CancelledError):
        await task
    release_write.set()
    await asyncio.gather(*fixture.service.pending_log_tasks)
    entry = fixture.uow.chat_logs.save.await_args.args[0]
    assert entry.outcome == AnswerOutcome.CANCELLED
    assert entry.answer == "partial"
    fixture.uow.messages.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_prompt_budget_error_code_and_failed_log_preserve_original(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fixture = chat_fixture()

    async def stream(**_kwargs: object) -> AsyncGenerator[StreamEvent]:
        yield PipelineMetaEvent(request_id="request")
        raise ContextBudgetExceededError

    fixture.rag.stream.side_effect = stream
    fixture.uow.chat_logs.save.side_effect = RuntimeError("secret-driver-parameter")
    with pytest.raises(ContextBudgetExceededError):
        await fixture.service.sync_chat("question", 7, 1, UserKind.INTERNAL, UserRole.USER)
    entry = fixture.uow.chat_logs.save.await_args.args[0]
    assert entry.outcome == AnswerOutcome.ERROR
    assert entry.error_code == "prompt_budget"
    assert "Failed to record chat attempt" in caplog.text
    assert "secret-driver-parameter" not in caplog.text


def test_snapshot_uses_content_and_excludes_keys_credentials_and_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = evidence_profile_scope(GenericEvidenceProfile())
    with scope:
        check_snapshot(monkeypatch)


def check_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    space = {"id": 1, "pack_key": "generic"}
    before_hash, before_data = config_snapshot(make_rag(), space)

    for name in ("llm_api_key", "embedding_api_key", "rerank_api_key"):
        monkeypatch.setattr(settings, name, SecretStr(f"test-secret-{name}"))

    monkeypatch.setattr(
        settings, "llm_base_url", "https://private-user:private-pass@example.com/v1?api_key=private-token"
    )
    after_hash, after_data = config_snapshot(make_rag(), space)
    assert before_hash == after_hash
    assert before_data == after_data
    serialized = repr(after_data).lower()
    assert not any(
        secret in serialized
        for secret in ("api_key", "test-secret", "private-user", "private-pass", "private-token")
    )
    assert "system_prompt" not in serialized
    assert (
        config_snapshot(make_rag(retriever=replace(make_rag().retriever, fetch_k=99)), space)[0] != after_hash
    )
    assert config_snapshot(make_rag(), {"id": 2, "pack_key": "generic"})[0] != after_hash


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["log", "snapshot"])
async def test_curator_detail_and_snapshot_enforce_managed_scope(endpoint: str) -> None:
    fixture = chat_fixture()
    assignment = Mock(
        get_scope=AsyncMock(return_value={"managed_client_ids": [10], "managed_internal_ids": [11]})
    )
    current = Mock(id=3, role=UserRole.CURATOR)
    fixture.uow.chat_logs.get_by_id.return_value = None
    fixture.uow.chat_logs.count_logs.return_value = 0

    if endpoint == "log":
        with pytest.raises(HTTPException) as exc:
            await get_chat_log(55, current, fixture.service._chat_log_service, assignment)
        fixture.uow.chat_logs.get_by_id.assert_awaited_once_with(55, (10, 11, 3))
    else:
        with pytest.raises(HTTPException) as exc:
            await get_config_snapshot("a" * 64, current, fixture.service._chat_log_service, assignment)
        assert fixture.uow.chat_logs.count_logs.await_args.args[0].user_ids == (10, 11, 3)

    assert exc.value.status_code == 404
    fixture.uow.chat_logs.get_snapshot.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [UserRole.ADMIN, UserRole.CURATOR])
async def test_visible_detail_contains_snapshot_and_diff(role: UserRole) -> None:
    fixture = chat_fixture()
    fixture.service._chat_log_service.current_snapshot = lambda _saved: {"rag": {"fetch_k": 50}}
    fixture.uow.chat_logs.get_by_id.return_value = ChatLog(id=5, user_id=10, space_id=1, config_hash="a" * 64)
    fixture.uow.chat_logs.get_snapshot.return_value = {"rag": {"fetch_k": 30}}
    assignment = Mock(
        get_scope=AsyncMock(return_value={"managed_client_ids": [10], "managed_internal_ids": []})
    )
    result = await get_chat_log(5, Mock(id=3, role=role), fixture.service._chat_log_service, assignment)
    assert result["entry"].id == 5
    assert result["snapshot"]["diff"] == [{"field": "rag.fetch_k", "saved": 30, "current": 50}]
    assert fixture.uow.chat_logs.get_by_id.await_args.args[1] == (
        (10, 3) if role == UserRole.CURATOR else None
    )


def test_aux_usage_counts_only_this_attempt_and_keeps_unknown_tokens() -> None:
    request = RagRequest(
        "q",
        (),
        RagContext(user=UserContext(1, UserKind.INTERNAL, UserRole.USER)),
        make_rag(),
        0,
        "request",
        "model",
        usage_start=1,
        usage_records=[
            {"operation": "llm", "input_tokens": 1000, "output_tokens": 1000},
            {"operation": "llm", "input_tokens": 10, "output_tokens": 2},
            {"operation": "generate", "input_tokens": 90, "output_tokens": 30},
            {"operation": "reranking", "input_tokens": 40, "output_tokens": 0},
        ],
    )
    assert auxiliary_usage(request).input_tokens == 10
    assert auxiliary_usage(request).output_tokens == 2
    assert auxiliary_tokens(request) == 12
    request.usage_records.append({"operation": "llm", "input_tokens": 3, "output_tokens": None})
    assert auxiliary_usage(request).input_tokens == 13
    assert auxiliary_usage(request).output_tokens is None
    assert auxiliary_tokens(request) is None
