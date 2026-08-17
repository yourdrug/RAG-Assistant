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
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import Breadth
from domain.value_objects.message_role import MessageRole
from domain.value_objects.roles import UserKind
from domain.value_objects.stream_events import MetaEvent, SourcesEvent, StreamEvent, TextChunk

from application.dto.chat_dto import ChatResult
from application.ports.chat_rag_port import ChatRAGPort
from application.ports.chat_support import RollingSummaryUpdaterPort
from application.ports.unit_of_work_factory import UnitOfWorkFactory

log = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        rag_service: ChatRAGPort,
        history_window: int = 8,
        rolling_summary_enabled: bool = False,
        summary_updater: RollingSummaryUpdaterPort | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._rag_service = rag_service
        self._history_window = history_window
        self._rolling_summary_enabled = rolling_summary_enabled
        self._summary_updater = summary_updater

    async def _get_user_context(self, user_id: int, user_kind: str) -> tuple[list[int], list[int]]:
        async with self._uow_factory.create() as uow:
            if user_kind == UserKind.CLIENT:
                return [], []
            group_ids = await uow.groups.get_user_group_ids(user_id)
            assigned_ids = await uow.client_assignments.get_assigned_client_ids(user_id)
            return group_ids or [], assigned_ids or []

    async def _save_chat_log(
        self,
        user_id: int,
        conversation_id: int,
        question: str,
        answer: str,
        sources: list[dict],
        latency_ms: int,
        breadth: str,
        domain: str,
        retrieval_count: int,
        reranker_score: float | None,
        model_used: str | None,
    ) -> None:
        try:
            chat_log = ChatLog(
                user_id=user_id,
                conversation_id=conversation_id,
                question=question,
                answer=answer,
                sources=sources,
                latency_ms=latency_ms,
                model_used=model_used,
                breadth=breadth,
                domain=domain,
                retrieval_count=retrieval_count,
                reranker_score=reranker_score,
            )
            async with self._uow_factory.create(master=True) as uow:
                await uow.chat_logs.save(chat_log)
        except Exception:
            log.exception("Failed to save chat log")

    async def stream_chat(
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        depth: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        group_ids, assigned_ids = await self._get_user_context(user_id, user_kind)
        ctx = ChatContext(
            user_id=user_id,
            user_kind=user_kind,
            user_group_ids=group_ids,
            assigned_client_ids=assigned_ids,
            depth=depth,
        )

        async with self._uow_factory.create() as uow:
            conv = await uow.conversations.get_or_create(conversation_id, user_id)

            history = await uow.messages.get_history(conv.id, window=self._history_window)
            if history and history[-1].role == MessageRole.USER:
                history = history[:-1]

        full_answer = ""
        sources: list[dict] = []
        confidence: float | None = None
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
            elif isinstance(event, TextChunk):
                if not user_msg_saved:
                    user_msg_saved = True
                    async with self._uow_factory.create() as uow:
                        user_msg = Message(
                            conversation_id=conv.id,
                            role=MessageRole.USER,
                            content=question,
                        )
                        await uow.messages.save(user_msg)

                full_answer += event.text
                yield event

        latency_ms = int((time.monotonic() - t_start) * 1000)

        async with self._uow_factory.create() as uow:
            msg_sources = list(sources) if sources else None
            if confidence is not None and msg_sources is not None:
                msg_sources = list(msg_sources)
                msg_sources.append({"_confidence": confidence})
            assistant_msg = Message(
                conversation_id=conv.id,
                role=MessageRole.ASSISTANT,
                content=full_answer,
                sources=msg_sources,
            )
            await uow.messages.save(assistant_msg)

        # Save chat log for quality tracking
        retrieval_count = len(sources)
        reranker_score = None
        if sources:
            scores = [s.get("max_score", 0) for s in sources if isinstance(s, dict)]
            reranker_score = max(scores) if scores else None

        await self._save_chat_log(
            user_id=user_id,
            conversation_id=conv.id,
            question=question,
            answer=full_answer,
            sources=sources,
            latency_ms=latency_ms,
            breadth=depth or Breadth.NARROW.value,
            domain=DocDomain.GENERAL.value,
            retrieval_count=retrieval_count,
            reranker_score=reranker_score,
            model_used=None,
        )

        # Rolling summary: fire-and-forget when history exceeds window
        if self._rolling_summary_enabled and len(history) >= self._history_window:
            recent_turns = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": full_answer},
            ]
            conv_id = conv.id
            updater = self._summary_updater

            async def _bg_update_summary() -> None:
                try:
                    async with self._uow_factory.create() as uow:
                        conv_model = await uow.conversations.get(conv_id)
                        if conv_model is not None and updater is not None:
                            existing = getattr(conv_model, "summary", None)
                            new_summary = await updater.update(existing, recent_turns)
                            conv_model.summary = new_summary
                            await uow.conversations.save(conv_model)
                except Exception:
                    log.exception("Failed to update rolling summary for conv %d", conv_id)

            asyncio.create_task(_bg_update_summary())

        yield MetaEvent(conversation_id=conv.id, sources=sources, confidence=confidence)

    async def sync_chat(
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        depth: str | None = None,
    ) -> ChatResult:
        group_ids, assigned_ids = await self._get_user_context(user_id, user_kind)
        ctx = ChatContext(
            user_id=user_id,
            user_kind=user_kind,
            user_group_ids=group_ids,
            assigned_client_ids=assigned_ids,
            depth=depth,
        )

        async with self._uow_factory.create() as uow:
            conv = await uow.conversations.get_or_create(conversation_id, user_id)

            if len(question.strip()) >= 3:
                user_msg = Message(
                    conversation_id=conv.id,
                    role=MessageRole.USER,
                    content=question,
                )
                await uow.messages.save(user_msg)

            history = await uow.messages.get_history(conv.id, window=self._history_window)
            if history and history[-1].role == MessageRole.USER:
                history = history[:-1]

        t_start = time.monotonic()
        rag_result = await self._rag_service.invoke(
            question=question,
            history=history,
            ctx=ctx,
        )
        latency_ms = int((time.monotonic() - t_start) * 1000)

        async with self._uow_factory.create() as uow:
            assistant_msg = Message(
                conversation_id=conv.id,
                role=MessageRole.ASSISTANT,
                content=rag_result.answer,
                sources=rag_result.sources,
            )
            await uow.messages.save(assistant_msg)

        # Save chat log for quality tracking
        await self._save_chat_log(
            user_id=user_id,
            conversation_id=conv.id,
            question=question,
            answer=rag_result.answer,
            sources=rag_result.sources,
            latency_ms=latency_ms,
            breadth=rag_result.breadth,
            domain=rag_result.domain,
            retrieval_count=rag_result.retrieval_count,
            reranker_score=rag_result.reranker_score,
            model_used=rag_result.model_used,
        )

        return ChatResult(answer=rag_result.answer, conversation_id=conv.id, sources=rag_result.sources)
