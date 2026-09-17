"""Application service for RAG chat orchestration.

Manages conversation lifecycle (create, list, delete), persists messages,
and streams RAG answers to the client via the ``ChatRAGPort``.  Each public
method opens its own async UnitOfWork via the injected UnitOfWorkFactory.
"""

from __future__ import annotations

import html
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date

from application.dto.chat_dto import ChatResult
from application.ports.chat_rag_port import ChatRAGPort
from application.ports.chat_settings import ChatSettingsPort
from application.ports.pii_redactor import PIIRedactorPort
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.chat_log_service import ChatLogService
from application.services.conversation_service import ConversationService
from application.services.user_context_factory import UserContextFactory
from domain.entities.conversation import Conversation
from domain.entities.message import Message
from domain.exceptions import BusinessRuleViolation
from domain.utils import compute_reranker_score
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.curator_scope import CuratorScope
from domain.value_objects.message_role import MessageRole
from domain.value_objects.stream_events import (
    MetaEvent,
    SourcesEvent,
    StatusEvent,
    StreamEvent,
    TextChunk,
    UsageReport,
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

    async def _prepare_chat(
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        depth: str | None,
        as_of_date: date | None,
    ) -> ChatSetup:
        async with self._uow_factory.create(master=True) as uow:
            user_ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role=user_role)
            conv = await uow.conversations.get_or_create(conversation_id, user_id)
            if conv.id is None:
                raise RuntimeError("Conversation get_or_create returned None id")
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
                raise BusinessRuleViolation(
                    f"Curator scope ({total_ids} ids) exceeds retrieval limit "
                    f"({self._settings.curator_scope_max_ids}); use group-based assignment"
                )

        pii_question = self._pii_redactor.redact(question)
        return ChatSetup(conv=conv, history=history, ctx=ctx, pii_question=pii_question)

    async def _persist_chat(
        self,
        conv_id: int,
        user_id: int,
        question: str,
        full_answer: str,
        sources: list[dict],
        latency_ms: int,
        depth: str | None,
        history: list,
        usage: UsageReport | None = None,
        model_used: str | None = None,
        domain: str | None = None,
    ) -> None:
        safe_question = html.escape(question)
        async with self._uow_factory.create(master=True) as uow:
            await uow.messages.save(
                Message(
                    conversation_id=conv_id,
                    role=MessageRole.USER,
                    content=safe_question,
                )
            )
            await uow.messages.save(
                Message(
                    conversation_id=conv_id,
                    role=MessageRole.ASSISTANT,
                    content=full_answer,
                    sources=sources,
                )
            )
            await self._chat_log_service.create(
                user_id=user_id,
                conv_id=conv_id,
                question=question,
                full_answer=full_answer,
                sources=sources,
                latency_ms=latency_ms,
                depth=depth,
                retrieval_count=len(sources),
                reranker_score=compute_reranker_score(sources),
                model_used=model_used,
                domain=domain,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                _existing_uow=uow,
            )

        if usage:
            log.info(
                "chat_token_usage conversation=%d input=%s output=%s model=%s",
                conv_id,
                usage.input_tokens,
                usage.output_tokens,
                usage.model,
            )

        await self._conversation_service.update_rolling_summary(conv_id, question, full_answer, history)

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
        depth: str | None = None,
        as_of_date: date | None = None,
    ) -> AsyncIterator[StreamEvent]:
        setup = await self._prepare_chat(
            question,
            conversation_id,
            user_id,
            user_kind,
            user_role,
            depth,
            as_of_date,
        )

        full_answer = ""
        sources: list[dict] = []
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
            raise RuntimeError("Conversation id is None during stream_chat")

        await self._persist_chat(
            conv_id=setup.conv.id,
            user_id=user_id,
            question=setup.pii_question,
            full_answer=full_answer,
            sources=sources,
            latency_ms=latency_ms,
            depth=depth,
            history=setup.history,
            usage=usage,
        )

        yield MetaEvent(conversation_id=setup.conv.id, sources=sources, confidence=confidence)

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
        depth: str | None = None,
        as_of_date: date | None = None,
    ) -> ChatResult:
        setup = await self._prepare_chat(
            question,
            conversation_id,
            user_id,
            user_kind,
            user_role,
            depth,
            as_of_date,
        )

        t_start = time.monotonic()
        rag_result = await self._rag_service.invoke(
            question=question,
            history=setup.history,
            ctx=setup.ctx,
        )
        latency_ms = int((time.monotonic() - t_start) * 1000)
        if setup.conv.id is None:
            raise RuntimeError("Conversation id is None during sync_chat")

        await self._persist_chat(
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
        )

        return ChatResult(
            answer=rag_result.answer,
            conversation_id=setup.conv.id,
            sources=rag_result.sources,
            confidence=rag_result.confidence,
            input_tokens=rag_result.input_tokens,
            output_tokens=rag_result.output_tokens,
        )
