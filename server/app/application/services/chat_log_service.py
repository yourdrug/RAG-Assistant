"""Application service for chat log creation and admin queries."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from application.ports.unit_of_work_factory import UnitOfWorkFactory
from domain.entities.chat_log import ChatLog
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import Breadth

if TYPE_CHECKING:
    from application.uow import UnitOfWork


class ChatLogService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def create(
        self,
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
        model_used: str | None = None,
        domain: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        _existing_uow: UnitOfWork | None = None,
    ) -> None:
        chat_log = ChatLog(
            user_id=user_id,
            conversation_id=conv_id,
            question=question,
            answer=full_answer,
            sources=sources,
            latency_ms=latency_ms,
            model_used=model_used,
            breadth=depth or Breadth.NARROW.value,
            domain=domain or DocDomain.GENERAL.value,
            retrieval_count=retrieval_count,
            reranker_score=reranker_score,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        if _existing_uow is not None:
            await _existing_uow.chat_logs.save(chat_log)
        else:
            async with self._uow_factory.create(master=True) as uow:
                await uow.chat_logs.save(chat_log)

    async def list_logs(
        self,
        user_id: int | None = None,
        domain: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        async with self._uow_factory.create() as uow:
            return await uow.chat_logs.list_logs(
                user_id=user_id,
                domain=domain,
                date_from=date_from,
                date_to=date_to,
                search=search,
                limit=limit,
                offset=offset,
            )

    async def count_logs(
        self,
        user_id: int | None = None,
        domain: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        search: str | None = None,
    ) -> int:
        async with self._uow_factory.create() as uow:
            return await uow.chat_logs.count_logs(
                user_id=user_id,
                domain=domain,
                date_from=date_from,
                date_to=date_to,
                search=search,
            )
