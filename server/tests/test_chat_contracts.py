"""Chat orchestration, scope and persistence contracts."""

from __future__ import annotations

from datetime import date
import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest

from application.chat import ChatLogService, ChatService, ConversationService
from domain.entities.chat_log import ChatLog
from domain.entities.conversation import Conversation
from domain.entities.message import Message
from domain.exceptions import BusinessRuleViolation, EntityNotFound, PermissionDeniedError
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import Breadth
from domain.value_objects.message_role import MessageRole
from domain.value_objects.rag_result import RagResult
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.stream_events import (
    MetaEvent,
    SourcesEvent,
    StatusEvent,
    TextChunk,
    UsageReport,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from domain.value_objects.chat_context import ChatContext
    from domain.value_objects.stream_events import (
        StreamEvent,
    )


def chat_fixture() -> SimpleNamespace:
    uow = AsyncMock()
    uow.conversations.get_or_create.return_value = Conversation(id=7, user_id=1, summary="summary")
    uow.messages.get_history.return_value = [
        Message(role=MessageRole.ASSISTANT, content="previous"),
        Message(role=MessageRole.USER, content="pending"),
    ]
    uow.groups.get_user_group_ids.return_value = [2]
    uow.assignments.get_managed_client_ids.return_value = [10]
    uow.assignments.get_managed_internal_ids.return_value = [11]
    uow.assignments.get_managed_group_ids.return_value = [12]
    factory = Mock()
    events = []

    async def enter() -> AsyncMock:
        events.append("enter")
        return uow

    async def leave(*_args: object) -> bool:
        events.append("exit")
        return False

    factory.create.return_value.__aenter__ = AsyncMock(side_effect=enter)
    factory.create.return_value.__aexit__ = AsyncMock(side_effect=leave)
    rag = Mock(invoke=AsyncMock(return_value=RagResult(answer="answer", confidence=0.8)))
    conversations = Mock(update_rolling_summary=AsyncMock())
    settings = SimpleNamespace(history_window=8, curator_scope_max_ids=3)
    service = ChatService(
        factory,
        rag,
        settings,
        chat_log_service=ChatLogService(factory),
        conversation_service=conversations,
        pii_redactor=Mock(redact=Mock(return_value="<safe>")),
    )
    return SimpleNamespace(
        service=service,
        uow=uow,
        factory=factory,
        rag=rag,
        conversations=conversations,
        settings=settings,
        events=events,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [UserRole.USER, UserRole.CURATOR])
async def test_sync_loads_scope_and_persists_after_rag(role: UserRole) -> None:
    fixture = chat_fixture()
    as_of = date(2025, 1, 2)

    async def invoke(question: str, history: list[Message], ctx: ChatContext) -> RagResult:
        assert fixture.events == ["enter", "exit"]
        assert ctx.user_group_ids == [2]
        assert ctx.as_of_date == as_of
        assert ctx.summary == "summary"
        assert [msg.content for msg in history] == ["previous"]
        assert question == "original"

        if role == UserRole.CURATOR:
            assert ctx.curator_scope is not None
            assert ctx.curator_scope.managed_client_ids == (10,)
            assert ctx.curator_scope.managed_internal_ids == (11,)
            assert ctx.curator_scope.managed_group_ids == (12,)
        else:
            assert ctx.curator_scope is None

        return RagResult(answer="answer", model_used="model", confidence=0.8, input_tokens=5)

    fixture.rag.invoke.side_effect = invoke
    result = await fixture.service.sync_chat(
        "original", 7, 1, UserKind.INTERNAL, role, depth="wide", as_of_date=as_of
    )

    assert result.answer == "answer"
    assert result.conversation_id == 7
    assert result.input_tokens == 5
    assert fixture.events == ["enter", "exit", "enter", "exit"]
    messages = [call.args[0] for call in fixture.uow.messages.save.await_args_list]
    assert [msg.content for msg in messages] == ["&lt;safe&gt;", "answer"]
    log = fixture.uow.chat_logs.save.await_args.args[0]
    assert log.question == "<safe>"
    assert log.model_used == "model"
    fixture.conversations.update_rolling_summary.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_preserves_order_and_usage_then_emits_meta() -> None:
    fixture = chat_fixture()
    usage = UsageReport(input_tokens=5, output_tokens=6, model="model")
    status = StatusEvent("retrieval")
    sources = [{"document_id": 3}]

    async def stream(**_kwargs: object) -> AsyncIterator[StreamEvent]:
        assert fixture.events == ["enter", "exit"]
        yield status
        yield TextChunk("an")
        yield SourcesEvent(sources, 0.7, usage)
        yield TextChunk("swer")

    fixture.rag.stream.side_effect = stream
    events = [
        event
        async for event in fixture.service.stream_chat(
            "original", 7, 1, UserKind.INTERNAL, UserRole.USER, depth="wide"
        )
    ]
    assert events == [status, TextChunk("an"), TextChunk("swer"), MetaEvent(7, sources, 0.7)]
    log = fixture.uow.chat_logs.save.await_args.args[0]
    assert log.answer == "answer"
    assert log.breadth == "wide"
    assert (log.input_tokens, log.output_tokens) == (5, 6)
    assert fixture.factory.create.call_count == 2


@pytest.mark.asyncio
async def test_excessive_curator_scope_stops_before_retrieval() -> None:
    fixture = chat_fixture()
    fixture.settings.curator_scope_max_ids = 2
    with pytest.raises(BusinessRuleViolation, match="exceeds retrieval limit"):
        await fixture.service.sync_chat("question", 7, 1, UserKind.INTERNAL, UserRole.CURATOR)
    fixture.rag.invoke.assert_not_awaited()
    fixture.uow.messages.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_rag_does_not_persist_messages_or_log() -> None:
    fixture = chat_fixture()
    fixture.rag.invoke.side_effect = RuntimeError("RAG failed")
    with pytest.raises(RuntimeError, match="RAG failed"):
        await fixture.service.sync_chat("question", 7, 1, UserKind.INTERNAL, UserRole.USER)
    fixture.uow.messages.save.assert_not_awaited()
    fixture.uow.chat_logs.save.assert_not_awaited()
    fixture.conversations.update_rolling_summary.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("role,allowed", [(UserRole.USER, False), (UserRole.ADMIN, True)])
async def test_conversation_history_checks_ownership(role: UserRole, *, allowed: bool) -> None:
    fixture = chat_fixture()
    fixture.uow.conversations.get_by_id.return_value = Conversation(id=7, user_id=2)
    service = ConversationService(fixture.factory, AsyncMock(), fixture.settings)

    if allowed:
        await service.get_history(7, 1, role)
        fixture.uow.messages.list_page.assert_awaited_once_with(7, limit=100, before_id=None)
    else:
        with pytest.raises(PermissionDeniedError):
            await service.get_history(7, 1, role)
        fixture.uow.messages.list_page.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_conversation_does_not_load_messages() -> None:
    fixture = chat_fixture()
    fixture.uow.conversations.get_by_id.return_value = None
    service = ConversationService(fixture.factory, AsyncMock(), fixture.settings)
    with pytest.raises(EntityNotFound):
        await service.get_history(7, 1, UserRole.USER)
    fixture.uow.messages.list_page.assert_not_awaited()


LEGACY_EXPORTS = [
    ("application.services.chat_service", ("ChatService", "ChatSetup")),
    ("application.services.chat_log_service", ("ChatLogService",)),
    ("application.services.conversation_service", ("ConversationService",)),
    ("application.dto.chat_dto", ("ChatResult",)),
    ("application.ports.chat_rag_port", ("ChatRAGPort",)),
    ("application.ports.chat_settings", ("ChatSettingsPort",)),
    ("application.ports.chat_support", ("RollingSummaryUpdaterPort",)),
    ("application.ports.pii_redactor", ("PIIRedactorPort",)),
]


@pytest.mark.parametrize("legacy,names", LEGACY_EXPORTS)
def test_legacy_exports_share_canonical_objects(legacy: str, names: tuple[str, ...]) -> None:
    old = importlib.import_module(legacy)
    canonical = importlib.import_module("application.chat")

    for name in names:
        assert getattr(old, name) is getattr(canonical, name)


@pytest.mark.asyncio
async def test_log_failure_uses_message_transaction_and_stops_summary() -> None:
    fixture = chat_fixture()
    fixture.uow.chat_logs.save.side_effect = RuntimeError("save failed")
    with pytest.raises(RuntimeError, match="save failed"):
        await fixture.service.sync_chat("question", 7, 1, UserKind.INTERNAL, UserRole.USER)
    assert fixture.factory.create.call_count == 2
    assert fixture.uow.messages.save.await_count == 2
    exit_args = fixture.factory.create.return_value.__aexit__.await_args.args
    assert exit_args[0] is RuntimeError
    fixture.conversations.update_rolling_summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_log_creation_without_existing_uow_opens_master_transaction() -> None:
    fixture = chat_fixture()
    await ChatLogService(fixture.factory).create(
        user_id=1,
        conv_id=7,
        question="question",
        full_answer="answer",
        sources=[],
        latency_ms=10,
        depth=None,
        retrieval_count=0,
        reranker_score=None,
    )
    fixture.factory.create.assert_called_once_with(master=True)
    log = fixture.uow.chat_logs.save.await_args.args[0]
    assert log.question == "question"
    assert log.answer == "answer"
    assert log.breadth == Breadth.NARROW.value
    assert log.domain == DocDomain.GENERAL.value


@pytest.mark.asyncio
async def test_log_listing_forwards_scope_and_loads_emails_in_same_uow() -> None:
    fixture = chat_fixture()
    log = ChatLog(user_id=2)
    fixture.uow.chat_logs.list_logs.return_value = [log]
    fixture.uow.users.get_by_ids.return_value = [SimpleNamespace(id=2, email="owner@example.test")]
    result = await ChatLogService(fixture.factory).list_logs_with_emails(user_ids=[2], limit=5, offset=3)
    assert result == ([log], {2: "owner@example.test"})
    fixture.uow.chat_logs.list_logs.assert_awaited_once_with(user_ids=[2], limit=5, offset=3)
    fixture.uow.users.get_by_ids.assert_awaited_once_with([2])
    fixture.factory.create.assert_called_once_with()


@pytest.mark.asyncio
async def test_failed_stream_does_not_persist_partial_answer() -> None:
    fixture = chat_fixture()

    async def stream(**_kwargs: object) -> AsyncIterator[StreamEvent]:
        yield TextChunk("partial")
        msg = "stream failed"
        raise RuntimeError(msg)

    fixture.rag.stream.side_effect = stream
    iterator = fixture.service.stream_chat("question", 7, 1, UserKind.INTERNAL, UserRole.USER)
    assert await anext(iterator) == TextChunk("partial")
    with pytest.raises(RuntimeError, match="stream failed"):
        await anext(iterator)
    fixture.uow.messages.save.assert_not_awaited()
    fixture.uow.chat_logs.save.assert_not_awaited()
    fixture.conversations.update_rolling_summary.assert_not_awaited()
