"""Application service for chat log creation and admin queries."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any

from application.rag.stream_events import PipelineMetaEvent
from domain.chat.chat_log import ChatLog
from domain.chat.repositories import DEFAULT_CHAT_LOG_FILTER, ChatLogFilter
from domain.rag.metadata import AnswerMetadata


if TYPE_CHECKING:
    from collections.abc import Callable

    from application.chat.dto import CreateChatLogCommand
    from application.uow import UnitOfWork, UnitOfWorkFactory


class ChatLogService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        current_snapshot: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self.current_snapshot = current_snapshot

    async def create(self, command: CreateChatLogCommand, *, uow: UnitOfWork | None = None) -> None:
        metadata = command.metadata or PipelineMetaEvent()
        chat_log = ChatLog(
            user_id=command.user_id,
            conversation_id=command.conv_id,
            question=command.question,
            answer=command.full_answer,
            sources=command.sources,
            latency_ms=command.latency_ms,
            model_used=command.model_used,
            breadth=command.depth,
            domain=command.domain,
            message_id=command.message_id,
            space_id=command.space_id,
            pack_key=command.pack_key,
            depth_requested=command.depth_requested,
            as_of_date=command.as_of_date,
            confidence=command.confidence,
            sources_count=len(command.sources),
            ttft_ms=int(metadata.ttft_sec * 1000) if metadata.ttft_sec is not None else None,
            generation_ms=int(metadata.generation_sec * 1000)
            if metadata.generation_sec is not None
            else None,
            **{field.name: getattr(metadata, field.name) for field in fields(AnswerMetadata)},
            input_tokens=command.input_tokens,
            output_tokens=command.output_tokens,
        )

        if uow is not None:
            if metadata.config_hash and metadata.config_data is not None:
                await uow.chat_logs.save_snapshot(metadata.config_hash, metadata.config_data)

            await uow.chat_logs.save(chat_log)
        else:
            async with self._uow_factory.create(master=True) as new_uow:
                if metadata.config_hash and metadata.config_data is not None:
                    await new_uow.chat_logs.save_snapshot(metadata.config_hash, metadata.config_data)

                await new_uow.chat_logs.save(chat_log)

    async def list_logs(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> list[ChatLog]:
        async with self._uow_factory.create() as uow:
            return await uow.chat_logs.list_logs(filters)

    async def count_logs(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> int:
        async with self._uow_factory.create() as uow:
            return await uow.chat_logs.count_logs(filters)

    async def list_logs_with_emails(
        self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER
    ) -> tuple[list[ChatLog], dict[int, str]]:
        async with self._uow_factory.create() as uow:
            logs = await uow.chat_logs.list_logs(filters)
            unique_ids = list({log.user_id for log in logs if log.user_id is not None})

            if unique_ids:
                users = await uow.users.get_by_ids(unique_ids)
                email_map = {u.id: u.email for u in users if u.id is not None}
            else:
                email_map = {}

            return logs, email_map

    async def get_log(
        self, log_id: int, user_ids: tuple[int, ...] | None = None
    ) -> tuple[ChatLog, dict[str, Any] | None] | None:
        async with self._uow_factory.create() as uow:
            entry = await uow.chat_logs.get_by_id(log_id, user_ids)

            if entry is None:
                return None

            snapshot = await uow.chat_logs.get_snapshot(entry.config_hash) if entry.config_hash else None
            return entry, snapshot

    async def get_snapshot(
        self, config_hash: str, user_ids: tuple[int, ...] | None = None
    ) -> dict[str, Any] | None:
        async with self._uow_factory.create() as uow:
            # A hash is not authorization: only expose snapshots referenced by a visible log.
            if not await uow.chat_logs.count_logs(ChatLogFilter(config_hash=config_hash, user_ids=user_ids)):
                return None

            return await uow.chat_logs.get_snapshot(config_hash)

    def compare_snapshot(self, snapshot: dict[str, Any] | None) -> dict[str, Any]:
        if snapshot is None:
            return {"data": None, "current": None, "diff": None}

        current = self.current_snapshot(snapshot) if self.current_snapshot is not None else None
        return {
            "data": snapshot,
            "current": current,
            "diff": snapshot_diff(snapshot, current) if current is not None else None,
        }


def snapshot_diff(saved: dict[str, Any], current: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    result = []

    for key in sorted(saved.keys() | current.keys()):
        old, new = saved.get(key), current.get(key)
        path = f"{prefix}.{key}" if prefix else key

        if isinstance(old, dict) and isinstance(new, dict):
            result.extend(snapshot_diff(old, new, path))
        elif old != new:
            result.append({"field": path, "saved": old, "current": new})

    return result
