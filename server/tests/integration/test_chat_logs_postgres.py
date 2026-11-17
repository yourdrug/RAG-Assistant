"""Atomic attempt persistence, scope filtering and snapshot deduplication in PostgreSQL."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker
from tests.integration.uow import SessionDatabase

from application.chat import ChatLogService, ChatService
from application.chat.dto import PersistChatCommand
from application.rag.stream_events import PipelineMetaEvent, UsageReport
from domain.chat.chat_log import ChatLog
from domain.chat.repositories import ChatLogFilter
from domain.rag.breadth import Breadth
from domain.rag.outcome import AnswerOutcome
from infrastructure.database.models import (
    ChatLogModel,
    ConversationModel,
    MessageModel,
    UserModel,
    rag_config_snapshots,
)
from infrastructure.repositories.conversation.sqlalchemy_chat_log_repository import (
    SQLAlchemyChatLogRepository,
)
from infrastructure.uow_factory import UnitOfWorkFactory


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


pytestmark = pytest.mark.integration


async def seed_conversation(engine: AsyncEngine) -> int:
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions.begin() as session:
        user = UserModel(email="owner@example.test", hashed_password="unused", role="user", kind="internal")  # noqa: S106 - disposable test user
        session.add(user)
        await session.flush()
        conversation = ConversationModel(user_id=user.id, space_id=1)
        session.add(conversation)
        await session.flush()
        return conversation.id


@pytest.mark.asyncio
async def test_persisted_message_log_and_snapshot_share_transaction(pg_engine: AsyncEngine) -> None:
    conv_id = await seed_conversation(pg_engine)
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    factory = UnitOfWorkFactory(SessionDatabase(sessions))
    service = ChatService(
        factory,
        Mock(),
        Mock(),
        chat_log_service=ChatLogService(factory),
        conversation_service=Mock(),
        pii_redactor=Mock(),
    )
    command = PersistChatCommand(
        conv_id=conv_id,
        user_id=1,
        question="question",
        full_answer="answer",
        sources=[],
        latency_ms=50,
        depth="broad",
        history=[],
        space_id=1,
        pack_key="generic",
        usage=UsageReport(10, 20),
        metadata=PipelineMetaEvent(
            outcome=AnswerOutcome.GROUNDED,
            breadth=Breadth.BROAD,
            config_hash="a" * 64,
            config_data={"rag": {"fetch_k": 30}},
        ),
    )
    message_id = await service.persist_chat(command)
    async with sessions() as session:
        row = (await session.execute(select(ChatLogModel))).scalar_one()
        assert row.message_id == message_id
        assert row.outcome == "grounded"
        assert await session.scalar(select(func.count()).select_from(MessageModel)) == 2
        assert await session.scalar(select(func.count()).select_from(rag_config_snapshots)) == 1
    assert command.metadata is not None
    broken = replace(command, metadata=replace(command.metadata, config_hash="b" * 64, config_data=None))
    # Missing snapshot violates the FK only after both messages have been flushed.
    with pytest.raises(IntegrityError, match="foreign key"):
        await service.persist_chat(broken)
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(MessageModel)) == 2
        assert await session.scalar(select(func.count()).select_from(ChatLogModel)) == 1


@pytest.mark.asyncio
async def test_concurrent_snapshots_and_log_filters(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)

    async def insert_snapshot() -> None:
        async with sessions.begin() as session:
            await SQLAlchemyChatLogRepository(session).save_snapshot("a" * 64, {"rag": {"fetch_k": 30}})

    await asyncio.gather(insert_snapshot(), insert_snapshot())
    async with sessions.begin() as session:
        repo = SQLAlchemyChatLogRepository(session)

        for latency in (10, 30, 20):
            await repo.save(
                ChatLog(
                    space_id=1,
                    outcome=AnswerOutcome.NOT_FOUND,
                    latency_ms=latency,
                    breadth="broad",
                    model_used="model",
                    config_hash="a" * 64,
                )
            )
    async with sessions() as session:
        repo = SQLAlchemyChatLogRepository(session)
        filters = ChatLogFilter(
            space_id=1,
            outcome="not_found",
            model="model",
            breadth="broad",
            config_hash="a" * 64,
            sort_by="latency",
            sort_order="asc",
            limit=2,
        )
        assert await repo.count_logs(filters) == 3
        assert [entry.latency_ms for entry in await repo.list_logs(filters)] == [10, 20]
        assert await repo.count_logs(replace(filters, space_id=999)) == 0
        assert await repo.count_logs(replace(filters, outcome="generated")) == 0
        assert await repo.count_logs(replace(filters, user_ids=())) == 0
        assert await session.scalar(select(func.count()).select_from(rag_config_snapshots)) == 1
        first = (await repo.list_logs())[0]
        assert first.id is not None
        assert await repo.get_by_id(first.id, ()) is None
