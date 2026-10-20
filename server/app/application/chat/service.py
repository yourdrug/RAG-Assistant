"""Application service for RAG chat orchestration.

Manages conversation lifecycle (create, list, delete), persists messages,
and streams RAG answers to the client via the ``ChatRAGPort``.  Each public
method opens its own async UnitOfWork via the injected UnitOfWorkFactory.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import logging
import time
from typing import TYPE_CHECKING, Any

from application.chat.dto import ChatResult, CreateChatLogCommand, PersistChatCommand
from application.identity import UserContextFactory
from domain.entities.message import Message
from domain.identity.user_context import CuratorScope
from domain.shared.exceptions import BusinessRuleViolation
from domain.utils import compute_reranker_score
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.message_role import MessageRole
from domain.value_objects.stream_events import (
    MetaEvent,
    SourcesEvent,
    StatusEvent,
    TextChunk,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import date

    from application.chat.conversations import ConversationService
    from application.chat.logs import ChatLogService
    from application.chat.ports import ChatRAGPort, ChatSettingsPort, PIIRedactorPort
    from application.uow import UnitOfWorkFactory
    from domain.entities.conversation import Conversation
    from domain.value_objects.stream_events import (
        StreamEvent,
    )


log = logging.getLogger(__name__)


@dataclass
class ChatSetup:
    conv: Conversation
    history: list[Message]
    ctx: ChatContext
    pii_question: str


class ChatService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        rag_service: ChatRAGPort,
        chat_settings: ChatSettingsPort,
        *,
        chat_log_service: ChatLogService,
        conversation_service: ConversationService,
        pii_redactor: PIIRedactorPort,
        user_ctx_factory: UserContextFactory | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._rag_service = rag_service
        self._settings = chat_settings
        self._chat_log_service = chat_log_service
        self._conversation_service = conversation_service
        self._pii_redactor = pii_redactor
        self._user_ctx_factory = user_ctx_factory or UserContextFactory()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def prepare_chat(
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        *,
        depth: str | None,
        as_of_date: date | None,
    ) -> ChatSetup:
        async with self._uow_factory.create(master=True) as uow:
            user_ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role=user_role)
            conv = await uow.conversations.get_or_create(conversation_id, user_id)

            if conv.id is None:
                msg = "Conversation get_or_create returned None id"
                raise RuntimeError(msg)

            history = await uow.messages.get_history(conv.id, window=self._settings.history_window)

            if history and history[-1].role == MessageRole.USER:
                history = history[:-1]

        ctx = ChatContext(
            user_id=user_id,
            user_kind=user_kind,
            user_role=user_role,
            user_group_ids=list(user_ctx.group_ids),
            curator_scope=(
                CuratorScope(
                    managed_client_ids=tuple(user_ctx.managed_client_ids),
                    managed_internal_ids=tuple(user_ctx.managed_internal_ids),
                    managed_group_ids=tuple(user_ctx.managed_group_ids),
                )
                if user_ctx.is_curator
                else None
            ),
            depth=depth,
            summary=conv.summary,
            as_of_date=as_of_date,
        )

        scope = ctx.curator_scope

        if scope is not None:
            total_ids = (
                len(scope.managed_client_ids) + len(scope.managed_internal_ids) + len(scope.managed_group_ids)
            )

            if total_ids > self._settings.curator_scope_max_ids:
                log.warning(
                    "Curator %d scope exceeds max: %d > %d",
                    user_id,
                    total_ids,
                    self._settings.curator_scope_max_ids,
                )
                msg = (
                    f"Curator scope ({total_ids} ids) exceeds retrieval limit "
                    f"({self._settings.curator_scope_max_ids}); use group-based assignment"
                )
                raise BusinessRuleViolation(msg)

        pii_question = self._pii_redactor.redact(question)
        return ChatSetup(conv=conv, history=history, ctx=ctx, pii_question=pii_question)

    async def persist_chat(self, command: PersistChatCommand) -> None:
        safe_question = html.escape(command.question)
        async with self._uow_factory.create(master=True) as uow:
            await uow.messages.save(
                Message(
                    conversation_id=command.conv_id,
                    role=MessageRole.USER,
                    content=safe_question,
                )
            )
            await uow.messages.save(
                Message(
                    conversation_id=command.conv_id,
                    role=MessageRole.ASSISTANT,
                    content=command.full_answer,
                    sources=command.sources,
                )
            )
            await self._chat_log_service.create(
                CreateChatLogCommand(
                    user_id=command.user_id,
                    conv_id=command.conv_id,
                    question=command.question,
                    full_answer=command.full_answer,
                    sources=command.sources,
                    latency_ms=command.latency_ms,
                    depth=command.depth,
                    retrieval_count=len(command.sources),
                    reranker_score=compute_reranker_score(command.sources),
                    model_used=command.model_used,
                    domain=command.domain,
                    input_tokens=command.usage.input_tokens if command.usage else None,
                    output_tokens=command.usage.output_tokens if command.usage else None,
                ),
                uow=uow,
            )

        if command.usage:
            log.info(
                "chat_token_usage conversation=%d input=%s output=%s model=%s",
                command.conv_id,
                command.usage.input_tokens,
                command.usage.output_tokens,
                command.usage.model,
            )

        await self._conversation_service.update_rolling_summary(
            command.conv_id, command.question, command.full_answer, command.history
        )

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    async def stream_chat(
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        *,
        depth: str | None = None,
        as_of_date: date | None = None,
    ) -> AsyncIterator[StreamEvent]:
        setup = await self.prepare_chat(
            question,
            conversation_id,
            user_id,
            user_kind,
            user_role,
            depth=depth,
            as_of_date=as_of_date,
        )

        full_answer = ""
        sources: list[dict[str, Any]] = []
        confidence: float | None = None
        usage = None
        t_start = time.monotonic()

        async for event in self._rag_service.stream(
            question=question,
            history=setup.history,
            ctx=setup.ctx,
        ):
            if isinstance(event, SourcesEvent):
                sources = event.sources
                confidence = event.confidence
                usage = event.usage
            elif isinstance(event, StatusEvent):
                yield event
            elif isinstance(event, TextChunk):
                full_answer += event.text
                yield event

        latency_ms = int((time.monotonic() - t_start) * 1000)

        if setup.conv.id is None:
            msg = "Conversation id is None during stream_chat"
            raise RuntimeError(msg)

        await self.persist_chat(
            PersistChatCommand(
                conv_id=setup.conv.id,
                user_id=user_id,
                question=setup.pii_question,
                full_answer=full_answer,
                sources=sources,
                latency_ms=latency_ms,
                depth=depth,
                history=setup.history,
                usage=usage,
            ),
        )

        yield MetaEvent(
            conversation_id=setup.conv.id,
            sources=sources,
            confidence=confidence,
            as_of_date=setup.ctx.as_of_date,
        )

    # ------------------------------------------------------------------
    # Synchronous chat
    # ------------------------------------------------------------------

    async def sync_chat(
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        *,
        depth: str | None = None,
        as_of_date: date | None = None,
    ) -> ChatResult:
        setup = await self.prepare_chat(
            question,
            conversation_id,
            user_id,
            user_kind,
            user_role,
            depth=depth,
            as_of_date=as_of_date,
        )

        t_start = time.monotonic()
        rag_result = await self._rag_service.invoke(
            question=question,
            history=setup.history,
            ctx=setup.ctx,
        )
        latency_ms = int((time.monotonic() - t_start) * 1000)

        if setup.conv.id is None:
            msg = "Conversation id is None during sync_chat"
            raise RuntimeError(msg)

        await self.persist_chat(
            PersistChatCommand(
                conv_id=setup.conv.id,
                user_id=user_id,
                question=setup.pii_question,
                full_answer=rag_result.answer,
                sources=rag_result.sources,
                latency_ms=latency_ms,
                depth=None,
                history=setup.history,
                usage=None,
                model_used=rag_result.model_used,
                domain=rag_result.domain,
            ),
        )

        return ChatResult(
            answer=rag_result.answer,
            conversation_id=setup.conv.id,
            sources=rag_result.sources,
            confidence=rag_result.confidence,
            input_tokens=rag_result.input_tokens,
            output_tokens=rag_result.output_tokens,
        )
