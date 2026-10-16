"""Application service for chat log creation and admin queries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

from domain.entities.chat_log import ChatLog
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import Breadth


if TYPE_CHECKING:
    from datetime import datetime

    from application.chat.dto import ChatLogCreateOptions, ChatLogListOptions
    from application.uow import UnitOfWorkFactory


class ChatLogService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def create(self, **options: Unpack[ChatLogCreateOptions]) -> None:
        user_id = options["user_id"]
        conv_id = options["conv_id"]
        question = options["question"]
        full_answer = options["full_answer"]
        sources = options["sources"]
        latency_ms = options["latency_ms"]
        depth = options["depth"]
        retrieval_count = options["retrieval_count"]
        reranker_score = options["reranker_score"]
        model_used = options.get("model_used")
        domain = options.get("domain")
        input_tokens = options.get("input_tokens")
        output_tokens = options.get("output_tokens")
        existing_uow = options.get("_existing_uow")
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

        if existing_uow is not None:
            await existing_uow.chat_logs.save(chat_log)
        else:
            async with self._uow_factory.create(master=True) as uow:
                await uow.chat_logs.save(chat_log)

    async def list_logs(self, **options: Unpack[ChatLogListOptions]) -> list[ChatLog]:
        async with self._uow_factory.create() as uow:
            return await uow.chat_logs.list_logs(**options)

    async def count_logs(
        self,
        *,
        user_id: int | None = None,
        user_ids: list[int] | None = None,
        domain: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        search: str | None = None,
    ) -> int:
        async with self._uow_factory.create() as uow:
            return await uow.chat_logs.count_logs(
                user_id=user_id,
                user_ids=user_ids,
                domain=domain,
                date_from=date_from,
                date_to=date_to,
                search=search,
            )

    async def list_logs_with_emails(
        self, **options: Unpack[ChatLogListOptions]
    ) -> tuple[list[ChatLog], dict[int, str]]:
        async with self._uow_factory.create() as uow:
            logs = await uow.chat_logs.list_logs(**options)
            unique_ids = list({log.user_id for log in logs if log.user_id is not None})

            if unique_ids:
                users = await uow.users.get_by_ids(unique_ids)
                email_map = {u.id: u.email for u in users if u.id is not None}
            else:
                email_map = {}

            return logs, email_map
