"""Dispatcher completion and reconciliation use the same durable outbox criterion."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker

from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus
from domain.knowledge.index_sync import OutboxStatus, VectorOutboxEntry
from infrastructure.database.models import VectorStoreOutboxModel
from infrastructure.repositories.config.sqlalchemy_vector_outbox_repository import (
    SQLAlchemyVectorOutboxRepository,
)
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
from infrastructure.uow_factory import UnitOfWorkFactory


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.mark.parametrize("status", list(OutboxStatus))
async def test_dispatcher_and_reconciler_require_every_entry_done(
    pg_engine: AsyncEngine,
    status: OutboxStatus,
) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        documents = SQLAlchemyDocumentRepository(session)
        outbox = SQLAlchemyVectorOutboxRepository(session)
        first = await documents.save(Document(filename="first.txt", status=DocumentStatus.INDEXING))
        second = await documents.save(Document(filename="second.txt", status=DocumentStatus.INDEXING))
        assert first.id is not None
        assert second.id is not None

        for document in (first, second):
            assert document.id is not None
            entry = await outbox.enqueue(VectorOutboxEntry(aggregate_id=document.id))
            assert entry.id is not None

            if status == OutboxStatus.DONE:
                await outbox.mark_done(entry.id)
            else:
                # Exercise all durable statuses, including dead letters and in-flight work.
                await session.execute(
                    update(VectorStoreOutboxModel)
                    .where(VectorStoreOutboxModel.id == entry.id)
                    .values(status=status.value)
                )

    async with sessions.begin() as session:
        documents = SQLAlchemyDocumentRepository(session)
        assert await documents.mark_done_if_indexing(first.id) == (status == OutboxStatus.DONE)
        assert await documents.reconcile_indexing_documents() == (
            [second.id] if status == OutboxStatus.DONE else []
        )


async def test_document_index_lock_serializes_independent_dispatchers(pg_engine: AsyncEngine) -> None:
    database = MagicMock()
    database.async_master_connection = pg_engine.connect
    first, second = UnitOfWorkFactory(database), UnitOfWorkFactory(database)
    attempted, acquired = asyncio.Event(), asyncio.Event()

    async def other_dispatcher() -> None:
        attempted.set()
        async with second.document_index_lock(42):
            acquired.set()

    async with asyncio.timeout(3):
        async with first.document_index_lock(42):
            task = asyncio.create_task(other_dispatcher())
            await attempted.wait()
            # Another document is independent even while the same-object writer waits.
            async with second.document_index_lock(43):
                assert not acquired.is_set()
        await task
        assert acquired.is_set()
