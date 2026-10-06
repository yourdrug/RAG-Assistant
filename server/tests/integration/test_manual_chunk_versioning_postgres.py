"""Edition inheritance, rollback and locking through real PostgreSQL transactions."""

from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker

from application.services.chunk_mutation_service import ChunkMutationService
from domain.entities.act_version import ActVersion
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.user_context import UserContext
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.database.models import ActVersionModel, ChunkModel, DocumentModel, VectorStoreOutboxModel
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository
from infrastructure.repositories.misc.sqlalchemy_act_version_repository import SQLAlchemyActVersionRepository
from infrastructure.uow_factory import UnitOfWorkFactory

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def seed_edition(sessions):
    async with sessions.begin() as session:
        doc = DocumentModel(
            filename="archived.pdf",
            visibility=DocumentVisibility.INTERNAL_PUBLIC.value,
            status=DocumentStatus.DONE,
        )
        session.add(doc)
        await session.flush()
        version = await SQLAlchemyActVersionRepository(session).create(
            ActVersion(None, None, doc.id, date(2020, 1, 1), date(2024, 1, 1), False)
        )
        return doc.id, version.id


def make_factory(sessions):
    database = MagicMock()
    database.get_session.side_effect = lambda **kwargs: sessions()
    return UnitOfWorkFactory(database)


async def test_manual_chunk_round_trip_and_later_version_updates(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    document_id, version_id = await seed_edition(sessions)
    service = ChunkMutationService(make_factory(sessions), SimpleNamespace(chunk_size=100), MagicMock())
    result = await service.add_chunk(document_id, "Manual regulatory passage. " * 4, 1, UserRole.ADMIN.value)
    user = UserContext(1, UserKind.INTERNAL, UserRole.ADMIN)
    async with sessions() as session:
        repo = SQLAlchemyChunkRepository(session)
        chunk = await repo.get_by_id(result.id)
        assert chunk.act_version_id == version_id
        assert chunk.effective_from == date(2020, 1, 1)
        assert chunk.effective_to == date(2024, 1, 1)
        assert chunk.is_current is False
        entry = (await session.execute(select(VectorStoreOutboxModel))).scalar_one()
        metadata = entry.payload["points"][0]["metadata"]
        assert metadata["act_version_id"] == version_id
        assert metadata["effective_from"] == "2020-01-01"
        assert metadata["effective_to"] == "2024-01-01"
        assert metadata["is_current"] is False
        assert await repo.search_substring("regulatory", user) == []
        historical = await repo.search_substring("regulatory", user, as_of_date=date(2022, 1, 1))
        assert [item.chunk_id for item in historical] == [result.id]
    async with sessions.begin() as session:
        # Same projection updates used by the version review/timeline services.
        repo = SQLAlchemyChunkRepository(session)
        assert await repo.set_current_by_act_version_ids([version_id], True) == 1
        assert await repo.update_temporal_by_act_version_id(version_id, date(2021, 1, 1), None) == 1
    async with sessions() as session:
        chunk = await SQLAlchemyChunkRepository(session).get_by_id(result.id)
        assert chunk.is_current is True
        assert chunk.effective_from == date(2021, 1, 1) and chunk.effective_to is None


async def test_outbox_failure_rolls_back_manual_chunk_and_document_stats(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    document_id, _ = await seed_edition(sessions)
    factory = make_factory(sessions)

    class FailingFactory:
        @asynccontextmanager
        async def create(self, master=False):
            async with factory.create(master=master) as uow:
                enqueue = uow.vector_outbox.enqueue

                async def fail_after_enqueue(entry):
                    await enqueue(entry)
                    raise RuntimeError("outbox failed")

                uow.vector_outbox.enqueue = AsyncMock(side_effect=fail_after_enqueue)
                yield uow

    index = MagicMock()
    service = ChunkMutationService(FailingFactory(), SimpleNamespace(chunk_size=100), index)
    with pytest.raises(RuntimeError, match="outbox failed"):
        await service.add_chunk(document_id, "Manual regulatory passage. " * 4, 1, UserRole.ADMIN.value)
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(ChunkModel)) == 0
        assert await session.scalar(select(func.count()).select_from(VectorStoreOutboxModel)) == 0
        doc = await session.get(DocumentModel, document_id)
        assert doc.has_manual_edits is False
    index.add.assert_not_called()


async def test_locked_edition_serializes_concurrent_date_update(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    document_id, version_id = await seed_edition(sessions)
    async with sessions.begin() as reader:
        await SQLAlchemyActVersionRepository(reader).get_by_document_id(document_id, for_update=True)
        async with sessions() as writer:
            await writer.execute(text("SET LOCAL lock_timeout = '100ms'"))
            with pytest.raises(DBAPIError, match="lock timeout"):
                await writer.execute(
                    update(ActVersionModel).where(ActVersionModel.id == version_id).values(is_current=True)
                )
            await writer.rollback()
    async with sessions.begin() as writer:
        await writer.execute(
            update(ActVersionModel).where(ActVersionModel.id == version_id).values(is_current=True)
        )


async def seed_linked_timeline(sessions):
    from infrastructure.database.models import RegulatoryActModel

    async with sessions.begin() as session:
        act = RegulatoryActModel(act_type="decree", act_number="5", title="Timeline correction")
        session.add(act)
        await session.flush()
        ids = []
        for index, start in enumerate([date(2024, 1, 1), date(2025, 1, 1)]):
            doc = DocumentModel(
                filename=f"timeline-{index}.pdf",
                visibility=DocumentVisibility.INTERNAL_PUBLIC.value,
                status=DocumentStatus.DONE,
            )
            session.add(doc)
            await session.flush()
            version = await SQLAlchemyActVersionRepository(session).create(
                ActVersion(None, act.id, doc.id, start, date(2025, 1, 1) if index == 0 else None, index == 1)
            )
            session.add(
                ChunkModel(
                    document_id=doc.id,
                    chunk_index=0,
                    content="Regulatory timeline passage",
                    visibility=DocumentVisibility.INTERNAL_PUBLIC.value,
                    act_version_id=version.id,
                    effective_from=version.effective_from,
                    effective_to=version.effective_to,
                    is_current=version.is_current,
                )
            )
            ids.append(version.id)
        return act.id, ids


@pytest.mark.parametrize("new_start", [date(2026, 1, 1), date(2028, 1, 1)])
async def test_date_correction_updates_postgres_chain_and_search(pg_engine, new_start):
    from application.services.act_versioning_service import ActVersioningService

    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    _, ids = await seed_linked_timeline(sessions)
    service = ActVersioningService(make_factory(sessions), MagicMock())
    service._commands._today = lambda: date(2026, 6, 1)
    await service.update_version(ids[1], effective_from=new_start)
    async with sessions() as session:
        versions = (await session.scalars(select(ActVersionModel).order_by(ActVersionModel.id))).all()
        assert versions[0].effective_to == new_start
        assert versions[0].is_current == (new_start > date(2026, 6, 1))
        assert versions[1].is_current != versions[0].is_current
        historical = await SQLAlchemyChunkRepository(session).search_substring(
            "timeline", UserContext(1, UserKind.INTERNAL, UserRole.ADMIN), as_of_date=date(2025, 6, 1)
        )
        assert [chunk.act_version_id for chunk in historical] == [ids[0]]
        assert await session.scalar(select(func.count()).select_from(VectorStoreOutboxModel)) == 2


async def test_concurrent_date_corrections_serialize_timeline(pg_engine):
    import asyncio

    from application.services.act_versioning_service import ActVersioningService

    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    act_id, ids = await seed_linked_timeline(sessions)
    service = ActVersioningService(make_factory(sessions), MagicMock())
    service._commands._today = lambda: date(2026, 6, 1)
    await asyncio.wait_for(
        asyncio.gather(
            service.update_version(ids[0], effective_from=date(2023, 1, 1)),
            service.update_version(ids[1], effective_from=date(2026, 1, 1)),
        ),
        timeout=10,
    )
    async with sessions() as session:
        versions = await SQLAlchemyActVersionRepository(session).list_by_act(act_id)
        assert versions[0].effective_from == date(2023, 1, 1)
        assert versions[0].effective_to == versions[1].effective_from == date(2026, 1, 1)
        assert not versions[0].is_current and versions[1].is_current
