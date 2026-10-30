"""Transitions refresh a stale ORM identity after a concurrent locked write."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker

from domain.benchmark.sweep import BenchmarkSweep, BenchmarkSweepStatus
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus
from domain.shared.exceptions import BusinessRuleViolationError
from infrastructure.database.models import BenchmarkSweepModel, DocumentModel
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_document_transition_rechecks_state_after_waiting_for_lock(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        document = await SQLAlchemyDocumentRepository(session).save(
            Document(filename="race.txt", status=DocumentStatus.PROCESSING)
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
