"""Only completed vector outbox work permits INDEXING reconciliation."""

import asyncio
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.knowledge.index_sync import OutboxOperation, OutboxStatus
from infrastructure.database.models import DocumentModel, VectorStoreOutboxModel
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_reconcile_checks_all_outbox_states_and_is_idempotent(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        documents = []

        for status in [None, *OutboxStatus]:
            document = DocumentModel(
                filename=f"{status}.txt",
                visibility=DocumentVisibility.INTERNAL_PUBLIC.value,
                status=DocumentStatus.INDEXING.value,
            )
            session.add(document)
            await session.flush()
            documents.append(document)

            if status is not None:
                session.add(
                    VectorStoreOutboxModel(
                        aggregate_type="document",
                        aggregate_id=document.id,
                        operation=OutboxOperation.UPSERT_CHUNKS.value,
                        status=status.value,
                        payload={},
                        next_attempt_at=datetime.now(UTC),
                    )
                )

        expected = {documents[0].id, documents[3].id}  # no entries, DONE
        # A different aggregate with the same ID must not block document reconciliation.
        session.add(
            VectorStoreOutboxModel(
                aggregate_type="act_version",
                aggregate_id=documents[0].id,
                operation=OutboxOperation.UPDATE_METADATA.value,
                status=OutboxStatus.PENDING.value,
                payload={},
                next_attempt_at=datetime.now(UTC),
            )
        )
    async with sessions.begin() as session:
        repository = SQLAlchemyDocumentRepository(session)
        assert set(await repository.reconcile_indexing_documents()) == expected
        assert await repository.reconcile_indexing_documents() == []
    async with sessions() as session:
        saved = (await session.scalars(select(DocumentModel))).all()

        for document in saved:
            if document.id in expected:
                assert document.status == DocumentStatus.DONE.value
                assert document.indexed_at is not None
                assert document.version == 1
            else:
                assert document.status == DocumentStatus.INDEXING.value
                assert document.indexed_at is None
                assert document.version == 0


@pytest.mark.asyncio
async def test_concurrent_reconciliation_skips_owned_rows_and_rollback_allows_retry(
    pg_engine: AsyncEngine,
) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        document = DocumentModel(
            filename="retry.txt",
            visibility=DocumentVisibility.INTERNAL_PUBLIC.value,
            status=DocumentStatus.INDEXING.value,
        )
        session.add(document)
    locked, release = asyncio.Event(), asyncio.Event()

    async def first() -> None:
        async with sessions() as session:
            assert await SQLAlchemyDocumentRepository(session).reconcile_indexing_documents() == [document.id]
            locked.set()
            await release.wait()
            await session.rollback()

    task = asyncio.create_task(first())
    try:
        await asyncio.wait_for(locked.wait(), 5)
        async with sessions.begin() as session:
            assert (
                await asyncio.wait_for(
                    SQLAlchemyDocumentRepository(session).reconcile_indexing_documents(), 5
                )
                == []
            )
    finally:
        release.set()
        await task
    async with sessions.begin() as session:
        assert await SQLAlchemyDocumentRepository(session).reconcile_indexing_documents() == [document.id]
