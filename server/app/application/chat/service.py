"""Application service for RAG chat orchestration.

Manages conversation lifecycle (create, list, delete), persists messages,
and streams RAG answers to the client via the ``ChatRAGPort``.  Each public
method opens its own async UnitOfWork via the injected UnitOfWorkFactory.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, field, replace
import html
import logging
import time
from typing import TYPE_CHECKING, Any

from application.chat.dto import ChatResult, CreateChatLogCommand, PersistChatCommand
from application.identity import UserContextFactory
from application.rag.stream_events import (
    MetaEvent,
    PipelineMetaEvent,
    SourcesEvent,
    StatusEvent,
    TextChunk,
)
from application.spaces.resolution import client_space, resolve_space
from domain.chat.message import Message, MessageRole
from domain.identity.roles import UserKind
from domain.rag.context import RagContext
from domain.rag.exceptions import ContextBudgetExceededError
from domain.rag.outcome import AnswerOutcome
from domain.shared.exceptions import (
    BusinessRuleViolationError,
    LLMUnavailableError,
    SemaphoreTimeoutError,
    ValidationError,
)


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from datetime import date

    from application.chat.conversations import ConversationService
    from application.chat.logs import ChatLogService
    from application.chat.ports import ChatRAGPort, ChatSettingsPort, PIIRedactorPort
    from application.rag.stream_events import (
        StreamEvent,
        UsageReport,
    )
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.chat.conversation import Conversation
    from domain.knowledge.space import KnowledgeSpace


log = logging.getLogger(__name__)


@dataclass
class ChatSetup:
    conv: Conversation
    history: list[Message]
    ctx: RagContext
    pii_question: str


@dataclass
class ChatAttempt:
    setup: ChatSetup
    conv_id: int
    user_id: int
    depth: str | None
    started: float = field(default_factory=time.monotonic)
    full_answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    usage: UsageReport | None = None
    metadata: PipelineMetaEvent = field(default_factory=PipelineMetaEvent)
    first_text_at: float | None = None

    def observe(self, event: StreamEvent) -> bool:
        if isinstance(event, SourcesEvent):
            self.sources, self.confidence, self.usage = event.sources, event.confidence, event.usage
        elif isinstance(event, PipelineMetaEvent):
            self.metadata = event
        elif isinstance(event, StatusEvent):
            return True
        elif isinstance(event, TextChunk):
            if self.first_text_at is None and event.text:
                self.first_text_at = time.monotonic()

            self.full_answer += event.text
            return True

        return False


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
        self.pending_log_tasks: set[asyncio.Task[None]] = set()
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

    async def prepare_chat(  # noqa: PLR0913 - keyword-only chat options of one request
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        *,
        depth: str | None,
        as_of_date: date | None,
        space_id: int | None = None,
    ) -> ChatSetup:
        async with self._uow_factory.create(master=True) as uow:
            user_ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role=user_role)
            conv = await uow.conversations.get_or_create(conversation_id, user_id)

            if conv.id is None:
                msg = "Conversation get_or_create returned None id"
                raise RuntimeError(msg)

            if user_ctx.user_kind == UserKind.CLIENT:
                assigned = await client_space(uow, user_id)

                if assigned is None:
                    msg = "No support space assigned to this client"
                    raise ValidationError(msg, field="space_id")

                if space_id is not None and space_id != assigned.id:
                    msg = "Clients cannot choose a knowledge space"
                    raise ValidationError(msg, field="space_id")

                space_id = assigned.id

            space = await self.conversation_space(uow, conv, space_id)

            history = await uow.messages.get_history(conv.id, window=self._settings.history_window)

            if history and history[-1].role == MessageRole.USER:
                history = history[:-1]

        ctx = RagContext(
            user=user_ctx,
            depth=depth,
            summary=conv.summary,
            as_of_date=as_of_date,
            space=space.ref,
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
                raise BusinessRuleViolationError(msg)

        pii_question = self._pii_redactor.redact(question)
        return ChatSetup(conv=conv, history=history, ctx=ctx, pii_question=pii_question)

    @staticmethod
    async def conversation_space(uow: UnitOfWork, conv: Conversation, space_id: int | None) -> KnowledgeSpace:
        """Every question of a conversation is asked in the space of its first question."""
        if conv.space_id is not None:
            if space_id is not None and space_id != conv.space_id:
                msg = "The conversation is bound to another knowledge space; start a new conversation"
                raise ValidationError(msg, field="space_id")

            return await resolve_space(uow, conv.space_id)

        space = await resolve_space(uow, space_id)

        if conv.id is not None and space.id is not None:
            bound_space_id = await uow.conversations.assign_space(conv.id, space.id)

            if bound_space_id != space.id:
                msg = "The conversation is bound to another knowledge space; start a new conversation"
                raise ValidationError(msg, field="space_id")

            conv.space_id = bound_space_id

        return space

    async def persist_chat(self, command: PersistChatCommand) -> int:
        async with self._uow_factory.create(master=True) as uow:
            await uow.messages.save(
                Message(
                    conversation_id=command.conv_id,
                    role=MessageRole.USER,
                    content=html.escape(command.question),
                )
            )
            assistant = Message(
                conversation_id=command.conv_id,
                role=MessageRole.ASSISTANT,
                content=command.full_answer,
                sources=command.sources,
            )
            await uow.messages.save(assistant)

            if assistant.id is None:
                msg = "Saved assistant message has no id"
                raise RuntimeError(msg)

            await self._chat_log_service.create(self.log_command(command, assistant.id), uow=uow)
        return assistant.id

    async def update_summary(self, command: PersistChatCommand) -> None:
        try:
            await self._conversation_service.update_rolling_summary(
                command.conv_id, command.question, command.full_answer, command.history
            )
        except Exception:  # noqa: BLE001 - committed chat must survive summary failure; omit private error text.
            log.warning("Rolling summary update failed conversation=%d", command.conv_id)
        except asyncio.CancelledError:
            # The transaction already committed: this remains a completed attempt.
            log.info("Summary update cancelled conversation=%d", command.conv_id)

    @staticmethod
    def log_command(command: PersistChatCommand, message_id: int | None = None) -> CreateChatLogCommand:
        return CreateChatLogCommand(
            user_id=command.user_id,
            conv_id=command.conv_id,
            question=command.question,
            full_answer=command.full_answer,
            sources=command.sources,
            latency_ms=command.latency_ms,
            depth=command.depth,
            depth_requested=command.depth_requested,
            space_id=command.space_id,
            pack_key=command.pack_key,
            message_id=message_id,
            as_of_date=command.as_of_date,
            confidence=command.confidence,
            metadata=command.metadata,
            model_used=command.model_used,
            domain=command.domain,
            input_tokens=command.usage.input_tokens if command.usage else None,
            output_tokens=command.usage.output_tokens if command.usage else None,
        )

    def attempt_command(self, attempt: ChatAttempt) -> PersistChatCommand:
        space = attempt.setup.ctx.space
        safe_meta = replace(
            attempt.metadata,
            condensed_query=self._pii_redactor.redact(attempt.metadata.condensed_query)
            if attempt.metadata.condensed_query
            else None,
            sub_queries=[self._pii_redactor.redact(q) for q in attempt.metadata.sub_queries]
            if attempt.metadata.sub_queries is not None
            else None,
            refined_query=self._pii_redactor.redact(attempt.metadata.refined_query)
            if attempt.metadata.refined_query
            else None,
            ttft_sec=(attempt.first_text_at - attempt.started)
            if attempt.first_text_at is not None
            else attempt.metadata.ttft_sec,
        )
        return PersistChatCommand(
            conv_id=attempt.conv_id,
            user_id=attempt.user_id,
            question=attempt.setup.pii_question,
            full_answer=self._pii_redactor.redact(attempt.full_answer),
            sources=attempt.sources,
            latency_ms=int((time.monotonic() - attempt.started) * 1000),
            depth=attempt.metadata.breadth.value if attempt.metadata.breadth else None,
            depth_requested=attempt.depth,
            history=attempt.setup.history,
            usage=attempt.usage,
            model_used=attempt.metadata.model_used or (attempt.usage.model if attempt.usage else None),
            domain=attempt.metadata.domain or None,
            space_id=space.id if space else 0,
            pack_key=space.pack_key if space else None,
            as_of_date=attempt.setup.ctx.as_of_date,
            confidence=attempt.confidence,
            metadata=safe_meta,
        )

    async def record_failed_attempt(self, command: PersistChatCommand) -> None:
        try:
            async with asyncio.timeout(5):
                await self._chat_log_service.create(self.log_command(command))
        except Exception:  # noqa: BLE001 - bounded best-effort logging must preserve the original failure.
            # Never include exception text: driver errors may contain private parameters.
            log.warning("Failed to record chat attempt conversation=%d", command.conv_id)

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    async def stream_chat(  # noqa: PLR0913 - keyword-only chat options of one request
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        *,
        depth: str | None = None,
        as_of_date: date | None = None,
        space_id: int | None = None,
    ) -> AsyncGenerator[StreamEvent]:
        setup = await self.prepare_chat(
            question,
            conversation_id,
            user_id,
            user_kind,
            user_role,
            depth=depth,
            as_of_date=as_of_date,
            space_id=space_id,
        )

        if setup.conv.id is None:
            msg = "Conversation id is required"
            raise RuntimeError(msg)

        attempt = ChatAttempt(setup=setup, conv_id=setup.conv.id, user_id=user_id, depth=depth)
        try:
            async with aclosing(
                self._rag_service.stream(question=question, history=setup.history, ctx=setup.ctx)
            ) as stream:
                async for event in stream:
                    if attempt.observe(event):
                        yield event
            message_id = await self.persist_chat(self.attempt_command(attempt))
        except (asyncio.CancelledError, GeneratorExit):
            attempt.metadata = replace(attempt.metadata, outcome=AnswerOutcome.CANCELLED)
            task = asyncio.create_task(self.record_failed_attempt(self.attempt_command(attempt)))
            self.pending_log_tasks.add(task)
            task.add_done_callback(self.pending_log_tasks.discard)
            await asyncio.shield(task)
            raise
        except Exception as exc:
            attempt.metadata = replace(
                attempt.metadata, outcome=AnswerOutcome.ERROR, error_code=chat_error_code(exc)
            )
            await self.record_failed_attempt(self.attempt_command(attempt))
            raise

        await self.update_summary(self.attempt_command(attempt))
        yield MetaEvent(
            conversation_id=attempt.conv_id,
            message_id=message_id,
            sources=attempt.sources,
            confidence=attempt.confidence,
            usage=attempt.usage,
            as_of_date=setup.ctx.as_of_date,
        )

    # ------------------------------------------------------------------
    # Synchronous chat
    # ------------------------------------------------------------------

    async def sync_chat(  # noqa: PLR0913 - keyword-only chat options of one request
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        *,
        depth: str | None = None,
        as_of_date: date | None = None,
        space_id: int | None = None,
    ) -> ChatResult:
        answer: list[str] = []
        final: MetaEvent | None = None
        async with aclosing(
            self.stream_chat(
                question,
                conversation_id,
                user_id,
                user_kind,
                user_role,
                depth=depth,
                as_of_date=as_of_date,
                space_id=space_id,
            )
        ) as stream:
            async for event in stream:
                if isinstance(event, TextChunk):
                    answer.append(event.text)
                elif isinstance(event, MetaEvent):
                    final = event

        if final is None:
            msg = "Chat stream completed without metadata"
            raise RuntimeError(msg)

        return ChatResult(
            answer="".join(answer),
            conversation_id=final.conversation_id,
            message_id=final.message_id,
            sources=final.sources,
            confidence=final.confidence,
            input_tokens=final.usage.input_tokens if final.usage else None,
            output_tokens=final.usage.output_tokens if final.usage else None,
        )


def chat_error_code(exc: Exception) -> str:
    if isinstance(exc, ContextBudgetExceededError):
        return "prompt_budget"

    if isinstance(exc, SemaphoreTimeoutError):
        return "server_busy"

    if isinstance(exc, (LLMUnavailableError, TimeoutError)):
        return "llm_unavailable"

    return "internal_error"
