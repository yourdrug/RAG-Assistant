"""Application service for chat log creation and admin queries."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any

from application.chat.review import ChatLogReviewService
from application.chat.traces import ChatTraceService
from application.rag.stream_events import PipelineMetaEvent
from domain.chat.chat_log import ChatLog
from domain.chat.repositories import DEFAULT_CHAT_LOG_FILTER, ChatLogFilter
from domain.rag.metadata import AnswerMetadata


if TYPE_CHECKING:
    from collections.abc import Callable

    from application.chat.dto import CreateChatLogCommand
    from application.chat.ports import ChatSettingsPort, PIIRedactorPort
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.chat.feedback import MessageFeedback
    from domain.chat.message import Message
    from domain.chat.review import ChatLogReview

EXPORT_MAX_ROWS = 5000
EXPORT_PAGE_SIZE = 500


@dataclass(frozen=True)
class ChatLogRow:
    entry: ChatLog
    email: str | None = None
    feedback: MessageFeedback | None = None
    review: ChatLogReview | None = None


@dataclass(frozen=True)
class ChatLogDetails:
    row: ChatLogRow
    snapshot: dict[str, Any] | None
    context: list[Message]


class ChatLogService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        current_snapshot: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        pii_redactor: PIIRedactorPort | None = None,
        chat_settings: ChatSettingsPort | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self.current_snapshot = current_snapshot
        self.traces = ChatTraceService(uow_factory)
        self.reviews = ChatLogReviewService(uow_factory, pii_redactor, chat_settings)

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

            await self.save_attempt(uow, chat_log, metadata)
        else:
            async with self._uow_factory.create(master=True) as new_uow:
                if metadata.config_hash and metadata.config_data is not None:
                    await new_uow.chat_logs.save_snapshot(metadata.config_hash, metadata.config_data)

                await self.save_attempt(new_uow, chat_log, metadata)

    @staticmethod
    async def save_attempt(uow: UnitOfWork, entry: ChatLog, metadata: PipelineMetaEvent) -> None:
        await uow.chat_logs.save(entry)

        if metadata.trace_data and entry.id is not None:
            await uow.chat_logs.save_trace(entry.id, metadata.trace_data)

    async def list_logs(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> list[ChatLog]:
        async with self._uow_factory.create() as uow:
            return await uow.chat_logs.list_logs(filters)

    async def count_logs(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> int:
        async with self._uow_factory.create() as uow:
            return await uow.chat_logs.count_logs(filters)

    async def list_rows(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> list[ChatLogRow]:
        async with self._uow_factory.create() as uow:
            return await self.rows(uow, await uow.chat_logs.list_logs(filters))

    async def stats(self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER) -> dict[str, Any]:
        async with self._uow_factory.create() as uow:
            return await uow.chat_logs.stats(filters)

    async def export_rows(
        self, filters: ChatLogFilter = DEFAULT_CHAT_LOG_FILTER, max_rows: int = EXPORT_MAX_ROWS
    ) -> list[ChatLogRow]:
        """Rows of the filter in its sort order, at most ``max_rows`` from ``filters.offset``."""
        result: list[ChatLogRow] = []

        async with self._uow_factory.create() as uow:
            while len(result) < max_rows:
                page = replace(
                    filters,
                    offset=filters.offset + len(result),
                    limit=min(EXPORT_PAGE_SIZE, max_rows - len(result)),
                )
                logs = await uow.chat_logs.list_logs(page)
                result.extend(await self.rows(uow, logs))

                if len(logs) < page.limit:
                    break

        return result

    @staticmethod
    async def rows(uow: UnitOfWork, logs: list[ChatLog]) -> list[ChatLogRow]:
        user_ids = list({log.user_id for log in logs if log.user_id is not None})
        users = await uow.users.get_by_ids(user_ids) if user_ids else []
        emails = {user.id: user.email for user in users if user.id is not None}
        message_ids = [log.message_id for log in logs if log.message_id is not None]
        feedback = {
            (item.message_id, item.user_id): item
            for item in await uow.message_feedback.list_for_messages(message_ids)
        }
        reviews = {
            item.chat_log_id: item
            for item in await uow.chat_logs.list_reviews([log.id for log in logs if log.id is not None])
        }
        return [
            ChatLogRow(
                entry=log,
                email=emails.get(log.user_id) if log.user_id is not None else None,
                # Only the asking user's reaction belongs to the entry.
                feedback=feedback.get((log.message_id, log.user_id))
                if log.message_id is not None and log.user_id is not None
                else None,
                review=reviews.get(log.id) if log.id is not None else None,
            )
            for log in logs
        ]

    async def get_log(self, log_id: int, user_ids: tuple[int, ...] | None = None) -> ChatLogDetails | None:
        async with self._uow_factory.create() as uow:
            entry = await uow.chat_logs.get_by_id(log_id, user_ids)

            if entry is None:
                return None

            snapshot = await uow.chat_logs.get_snapshot(entry.config_hash) if entry.config_hash else None
            [row] = await self.rows(uow, [entry])
            return ChatLogDetails(
                row=row, snapshot=snapshot, context=await self.reviews.previous_turns(uow, entry)
            )

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
