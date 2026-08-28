"""Application service for RAG chat orchestration.

Manages conversation lifecycle (create, list, delete), persists messages,
and streams RAG answers to the client via the ``ChatRAGPort``.  Each public
method opens its own async UnitOfWork via the injected UnitOfWorkFactory.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from domain.entities.chat_log import ChatLog
from domain.entities.message import Message
from domain.utils import compute_reranker_score
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import Breadth
from domain.value_objects.message_role import MessageRole
from domain.value_objects.roles import UserKind
from domain.value_objects.stream_events import MetaEvent, SourcesEvent, StatusEvent, StreamEvent, TextChunk

from application.dto.chat_dto import ChatResult
from application.ports.chat_rag_port import ChatRAGPort
from application.ports.chat_settings import ChatSettingsPort
from application.ports.chat_support import RollingSummaryUpdaterPort
from application.ports.unit_of_work_factory import UnitOfWorkFactory

log = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        rag_service: ChatRAGPort,
        chat_settings: ChatSettingsPort,
        summary_updater: RollingSummaryUpdaterPort | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._rag_service = rag_service
        self._settings = chat_settings
        self._summary_updater = summary_updater
        self._background_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Managed background tasks
    # ------------------------------------------------------------------

    def _spawn_background(self, coro) -> None:
        """Spawn a background task with proper lifecycle tracking."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def shutdown(self) -> None:
        """Cancel all background tasks and wait for completion."""
        for t in self._background_tasks:
            t.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_user_context(self, user_id: int, user_kind: str) -> list[int]:
        async with self._uow_factory.create() as uow:
            if user_kind == UserKind.CLIENT:
                return []
            group_ids = await uow.groups.get_user_group_ids(user_id)
            return group_ids or []

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    async def _save_user_message(self, conv_id: int, question: str) -> None:
        async with self._uow_factory.create(master=True) as uow:
            user_msg = Message(
                conversation_id=conv_id,
                role=MessageRole.USER,
                content=question,
            )
            await uow.messages.save(user_msg)

    @staticmethod
    def _build_chat_log(
        *,
        user_id: int,
        conv_id: int,
        question: str,
        full_answer: str,
        sources: list[dict],
        latency_ms: int,
        depth: str | None,
        retrieval_count: int,
        reranker_score: float | None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> ChatLog:
        return ChatLog(
            user_id=user_id,
            conversation_id=conv_id,
            question=question,
            answer=full_answer,
            sources=sources,
            latency_ms=latency_ms,
            model_used=None,
            breadth=depth or Breadth.NARROW.value,
            domain=DocDomain.GENERAL.value,
            retrieval_count=retrieval_count,
            reranker_score=reranker_score,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def _handle_rolling_summary(
        self, conv_id: int, question: str, full_answer: str, history: list
    ) -> None:
        if not (self._settings.rolling_summary_enabled and len(history) >= self._settings.history_window):
            return
        recent_turns = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": full_answer},
        ]
        updater = self._summary_updater

        async def _bg_update_summary() -> None:
            try:
                async with self._uow_factory.create(master=True) as uow:
                    conv_model = await uow.conversations.get(conv_id)
                    if conv_model is not None and updater is not None:
                        existing = getattr(conv_model, "summary", None)
                        new_summary = await updater.update(existing, recent_turns)
                        conv_model.summary = new_summary
                        await uow.conversations.save(conv_model)
            except Exception:
                log.exception("Failed to update rolling summary for conv %d", conv_id)

        self._spawn_background(_bg_update_summary())

    async def stream_chat(
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        depth: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        group_ids = await self._get_user_context(user_id, user_kind)
        ctx = ChatContext(
            user_id=user_id,
            user_kind=user_kind,
            user_group_ids=group_ids,
            depth=depth,
        )

        # UoW 1: get/create conversation + history (get_or_create may INSERT)
        async with self._uow_factory.create(master=True) as uow:
            conv = await uow.conversations.get_or_create(conversation_id, user_id)
            assert conv.id is not None
            history = await uow.messages.get_history(conv.id, window=self._settings.history_window)
            if history and history[-1].role == MessageRole.USER:
                history = history[:-1]

        full_answer = ""
        sources: list[dict] = []
        confidence: float | None = None
        usage = None
        user_msg_saved = False
        t_start = time.monotonic()

        async for event in self._rag_service.stream(
            question=question,
            history=history,
            ctx=ctx,
        ):
            if isinstance(event, SourcesEvent):
                sources = event.sources
                confidence = event.confidence
                usage = event.usage
            elif isinstance(event, StatusEvent):
                yield event
            elif isinstance(event, TextChunk):
                if not user_msg_saved:
                    user_msg_saved = True
                    await self._save_user_message(conv.id, question)

                full_answer += event.text
                yield event

        latency_ms = int((time.monotonic() - t_start) * 1000)

        # UoW 2: atomic — save assistant message + chat log together
        retrieval_count = len(sources)
        reranker_score = compute_reranker_score(sources)

        async with self._uow_factory.create(master=True) as uow:
            assistant_msg = Message(
                conversation_id=conv.id,
                role=MessageRole.ASSISTANT,
                content=full_answer,
                sources=sources,
            )
            await uow.messages.save(assistant_msg)

            chat_log = self._build_chat_log(
                user_id=user_id,
                conv_id=conv.id,
                question=question,
                full_answer=full_answer,
                sources=sources,
                latency_ms=latency_ms,
                depth=depth,
                retrieval_count=retrieval_count,
                reranker_score=reranker_score,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
            )
            await uow.chat_logs.save(chat_log)

        if usage:
            log.info(
                "chat_token_usage conversation=%d input=%s output=%s model=%s",
                conv.id,
                usage.input_tokens,
                usage.output_tokens,
                usage.model,
            )

        await self._handle_rolling_summary(conv.id, question, full_answer, history)

        yield MetaEvent(conversation_id=conv.id, sources=sources, confidence=confidence)

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
    ) -> ChatResult:
        group_ids = await self._get_user_context(user_id, user_kind)
        ctx = ChatContext(
            user_id=user_id,
            user_kind=user_kind,
            user_group_ids=group_ids,
            depth=depth,
        )

        # UoW 1: get/create conversation + history (get_or_create may INSERT)
        async with self._uow_factory.create(master=True) as uow:
            conv = await uow.conversations.get_or_create(conversation_id, user_id)
            assert conv.id is not None
            history = await uow.messages.get_history(conv.id, window=self._settings.history_window)
            if history and history[-1].role == MessageRole.USER:
                history = history[:-1]

        # I/O: call LLM (no transaction needed)
        t_start = time.monotonic()
        rag_result = await self._rag_service.invoke(
            question=question,
            history=history,
            ctx=ctx,
        )
        latency_ms = int((time.monotonic() - t_start) * 1000)

        # UoW 2: atomic — save user msg + assistant msg + chat log together
        async with self._uow_factory.create(master=True) as uow:
            if len(question.strip()) >= 3:
                user_msg = Message(
                    conversation_id=conv.id,
                    role=MessageRole.USER,
                    content=question,
                )
                await uow.messages.save(user_msg)

            assistant_msg = Message(
                conversation_id=conv.id,
                role=MessageRole.ASSISTANT,
                content=rag_result.answer,
                sources=rag_result.sources,
            )
            await uow.messages.save(assistant_msg)

            chat_log = ChatLog(
                user_id=user_id,
                conversation_id=conv.id,
                question=question,
                answer=rag_result.answer,
                sources=rag_result.sources,
                latency_ms=latency_ms,
                model_used=rag_result.model_used,
                breadth=rag_result.breadth,
                domain=rag_result.domain,
                retrieval_count=rag_result.retrieval_count,
                reranker_score=rag_result.reranker_score,
                input_tokens=rag_result.input_tokens,
                output_tokens=rag_result.output_tokens,
            )
            await uow.chat_logs.save(chat_log)

        return ChatResult(
            answer=rag_result.answer,
            conversation_id=conv.id,
            sources=rag_result.sources,
            input_tokens=rag_result.input_tokens,
            output_tokens=rag_result.output_tokens,
        )
