"""Transitions refresh a stale ORM identity after a concurrent locked write."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker
from tests.fakes import DEFAULT_SPACE_ID

from application.chat.service import ChatService
from domain.benchmark.sweep import BenchmarkSweep, BenchmarkSweepStatus
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus
from domain.shared.exceptions import BusinessRuleViolationError, ValidationError
from infrastructure.database.models import BenchmarkSweepModel, DocumentModel, KnowledgeSpaceModel, UserModel
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)
from infrastructure.repositories.conversation.sqlalchemy_conversation_repository import (
    SQLAlchemyConversationRepository,
)
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
from infrastructure.repositories.space.sqlalchemy_space_repository import SQLAlchemyKnowledgeSpaceRepository


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from application.uow import UnitOfWork

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
TEST_HASH = "test"


@pytest.mark.parametrize("same_space", [False, True])
async def test_concurrent_first_questions_observe_the_committed_space(  # noqa: PLR0915 - coordinates two transactions and lock handoff
    pg_engine: AsyncEngine, same_space: bool
) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        user = UserModel(
            email="space-race@example.org", hashed_password=TEST_HASH, role="user", kind="internal"
        )
        other = KnowledgeSpaceModel(name="Other", pack_key="generic")
        session.add_all([user, other])
        await session.flush()
        conversation = await SQLAlchemyConversationRepository(session).create(user.id)
        other_id = other.id
    assert conversation.id is not None

    async with sessions() as writer, sessions() as reader:
        first_repo = SQLAlchemyConversationRepository(writer)
        second_repo = SQLAlchemyConversationRepository(reader)
        first = await first_repo.get_by_id(conversation.id)
        second = await second_repo.get_by_id(conversation.id)
        assert first is not None
        assert second is not None
        assert first.space_id is None
        assert second.space_id is None
        first_uow = cast(
            "UnitOfWork",
            SimpleNamespace(
                conversations=first_repo, knowledge_spaces=SQLAlchemyKnowledgeSpaceRepository(writer)
            ),
        )
        second_uow = cast(
            "UnitOfWork",
            SimpleNamespace(
                conversations=second_repo, knowledge_spaces=SQLAlchemyKnowledgeSpaceRepository(reader)
            ),
        )
        await ChatService.conversation_space(first_uow, first, DEFAULT_SPACE_ID)
        started = asyncio.Event()

        def notify(statement: str, **context: object) -> None:
            if statement.startswith("UPDATE conversations"):
                started.set()

        event.listen(pg_engine.sync_engine, "before_cursor_execute", notify, named=True)
        competing = asyncio.create_task(
            ChatService.conversation_space(second_uow, second, DEFAULT_SPACE_ID if same_space else other_id)
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            await writer.commit()

            if same_space:
                assert (await asyncio.wait_for(competing, timeout=5)).id == DEFAULT_SPACE_ID
            else:
                with pytest.raises(ValidationError):
                    await asyncio.wait_for(competing, timeout=5)
        finally:
            event.remove(pg_engine.sync_engine, "before_cursor_execute", notify)
            await reader.rollback()

            if not competing.done():
                competing.cancel()
                await asyncio.gather(competing, return_exceptions=True)

    async with sessions() as session:
        stored = await SQLAlchemyConversationRepository(session).get_by_id(conversation.id)
        assert stored is not None
        assert stored.space_id == DEFAULT_SPACE_ID


async def test_document_transition_rechecks_state_after_waiting_for_lock(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        document = await SQLAlchemyDocumentRepository(session).save(
            Document(space_id=DEFAULT_SPACE_ID, filename="race.txt", status=DocumentStatus.PROCESSING)
        )
    assert document.id is not None
    async with sessions() as stale_session, sessions() as writer:
        stale = await stale_session.get(DocumentModel, document.id)
        assert stale is not None
        await SQLAlchemyDocumentRepository(writer).update_status(document.id, status=DocumentStatus.DONE)
        started = asyncio.Event()

        def notify(statement: str, **context: object) -> None:
            if "FOR UPDATE" in statement:
                started.set()

        event.listen(pg_engine.sync_engine, "before_cursor_execute", notify, named=True)
        task = asyncio.create_task(
            SQLAlchemyDocumentRepository(stale_session).update_status(
                document.id, status=DocumentStatus.FAILED
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            await writer.commit()
            with pytest.raises(BusinessRuleViolationError):
                await task
            assert stale.status == DocumentStatus.DONE
            await stale_session.rollback()
        finally:
            event.remove(pg_engine.sync_engine, "before_cursor_execute", notify)
    async with sessions() as session:
        stored = await SQLAlchemyDocumentRepository(session).get_by_id(document.id)
        assert stored is not None
        assert stored.status == DocumentStatus.DONE


async def test_sweep_completion_cannot_overwrite_concurrent_cancellation(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        sweep = await SQLAlchemyBenchmarkSweepRepository(session).create(
            BenchmarkSweep(status=BenchmarkSweepStatus.RUNNING)
        )
    assert sweep.id is not None
    async with sessions() as stale_session, sessions() as writer:
        stale = await stale_session.get(BenchmarkSweepModel, sweep.id)
        assert stale is not None
        await SQLAlchemyBenchmarkSweepRepository(writer).update_status(
            sweep.id, BenchmarkSweepStatus.CANCELLED
        )
        started = asyncio.Event()

        def notify(statement: str, **context: object) -> None:
            if "FOR UPDATE" in statement:
                started.set()

        event.listen(pg_engine.sync_engine, "before_cursor_execute", notify, named=True)
        task = asyncio.create_task(
            SQLAlchemyBenchmarkSweepRepository(stale_session).update_status(
                sweep.id, BenchmarkSweepStatus.DONE
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            await writer.commit()
            with pytest.raises(BusinessRuleViolationError):
                await task
            assert stale.status == BenchmarkSweepStatus.CANCELLED
            await stale_session.rollback()
        finally:
            event.remove(pg_engine.sync_engine, "before_cursor_execute", notify)
