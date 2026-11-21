"""Sync channels on PostgreSQL: persistence, document bindings and the constraints behind them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker
from tests.fakes import DEFAULT_SPACE_ID
from tests.integration.uow import session_factory

from application.sync.channels import unlink_document
from domain.knowledge.document import Document, SyncBinding
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.shared.exceptions import BusinessRuleViolationError, UniqueConstraintError
from domain.sync.channel import ChannelAudience, ChannelSettings, GitHubLocation, SyncChannel
from domain.sync.run import FileOutcome, FileOutcomeKind, SyncRun, SyncRunStatus, SyncTrigger
from infrastructure.database.models import UserModel
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
from infrastructure.repositories.space.sqlalchemy_space_repository import SQLAlchemyKnowledgeSpaceRepository
from infrastructure.repositories.sync.sqlalchemy_sync_repository import (
    SQLAlchemySyncChannelRepository,
    SQLAlchemySyncRunRepository,
)


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def seed_channel(sessions: async_sessionmaker[AsyncSession]) -> SyncChannel:
    async with sessions.begin() as session:
        user = UserModel(email="owner@example.com", hashed_password="x", role="user", kind="internal")  # noqa: S106 - fixture user
        session.add(user)
        await session.flush()
        return await SQLAlchemySyncChannelRepository(session).add(
            SyncChannel(
                space_id=DEFAULT_SPACE_ID,
                settings=ChannelSettings(
                    location=GitHubLocation("ghe.example.com", 42, "acme/docs", "main"),
                    audience=ChannelAudience(DocumentVisibility.INTERNAL_PUBLIC),
                    responsible_user_id=user.id,
                    folder="docs",
                    exclusions=("drafts/",),
                ),
            )
        )


def bound(channel_id: int, path: str) -> Document:
    return Document(
        filename=path,
        space_id=DEFAULT_SPACE_ID,
        status=DocumentStatus.DONE,
        sync=SyncBinding(channel_id, path, "blob-sha", "commit-sha"),
    )


async def test_channel_and_run_round_trip(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    channel = await seed_channel(sessions)
    assert channel.id is not None

    with pytest.raises(BusinessRuleViolationError, match="sync channels"):
        async with sessions.begin() as session:
            await SQLAlchemyKnowledgeSpaceRepository(session).delete(DEFAULT_SPACE_ID)

    async with sessions.begin() as session:
        channels = SQLAlchemySyncChannelRepository(session)
        stored = await channels.get_by_id(channel.id, for_update=True)
        assert stored == channel
        assert stored is not None
        stored.change_settings(
            ChannelSettings(
                location=stored.settings.location,
                audience=stored.settings.audience,
                responsible_user_id=stored.settings.responsible_user_id,
                folder="docs/api",
                exclusions=("drafts/", "*.tmp.md"),
            ),
            user_id=0,
            is_admin=True,
        )
        await channels.update(stored)

    async with sessions.begin() as session:
        [listed] = await SQLAlchemySyncChannelRepository(session).list_by_space(DEFAULT_SPACE_ID)
        assert (listed.settings.folder, listed.settings.exclusions, listed.generation) == (
            "docs/api",
            ("drafts/", "*.tmp.md"),
            2,
        )
        runs = SQLAlchemySyncRunRepository(session)
        run = await runs.add(SyncRun(channel_id=channel.id, generation=2, trigger=SyncTrigger.WEBHOOK))
        run.start("c0ffee")
        run.complete(
            (
                FileOutcome("docs/api/a.md", FileOutcomeKind.ADDED),
                FileOutcome("b.md", FileOutcomeKind.FAILED, "x"),
            )
        )
        await runs.update(run)

    async with sessions.begin() as session:
        [latest] = await SQLAlchemySyncRunRepository(session).list_for_channel(channel.id)
        assert (latest.status, latest.storage_revision, latest.counts()) == (
            SyncRunStatus.PARTIAL,
            "c0ffee",
            {"added": 1, "failed": 1},
        )
        assert latest.outcomes[1].detail == "x"


async def test_document_bindings_are_unique_and_take_no_filename_slot(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    channel = await seed_channel(sessions)
    assert channel.id is not None

    async with sessions.begin() as session:
        documents = SQLAlchemyDocumentRepository(session)
        saved = await documents.save(bound(channel.id, "docs/a.md"))
        # A manual document with the same name and audience is a different slot.
        await documents.save(
            Document(filename="docs/a.md", space_id=DEFAULT_SPACE_ID, status=DocumentStatus.DONE)
        )
        assert (
            await documents.find_active_slot(None, "docs/a.md", None, space_id=DEFAULT_SPACE_ID) is not None
        )
        stored = await documents.get_by_id(saved.id or 0)
        assert stored is not None
        assert stored.sync == SyncBinding(channel.id, "docs/a.md", "blob-sha", "commit-sha")
        [listed] = await documents.list_channel_documents(channel.id)
        assert listed.id == saved.id

    with pytest.raises(UniqueConstraintError):
        async with sessions.begin() as session:
            await SQLAlchemyDocumentRepository(session).save(bound(channel.id, "docs/a.md"))


async def test_channel_with_bound_documents_cannot_be_dropped_until_unlinked(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    channel = await seed_channel(sessions)
    assert channel.id is not None

    async with sessions.begin() as session:
        saved = await SQLAlchemyDocumentRepository(session).save(bound(channel.id, "docs/a.md"))

    with pytest.raises(IntegrityError):
        async with sessions.begin() as session:
            await SQLAlchemySyncChannelRepository(session).delete(channel.id)

    async with sessions.begin() as session:
        await SQLAlchemyDocumentRepository(session).unlink_document(saved.id or 0, None)
        await SQLAlchemySyncChannelRepository(session).delete(channel.id)

    async with sessions.begin() as session:
        document = await SQLAlchemyDocumentRepository(session).get_by_id(saved.id or 0)
        assert document is not None
        assert document.sync is None


async def test_unlink_takes_a_free_name_when_a_manual_document_holds_it(pg_engine: AsyncEngine) -> None:
    """Without the rename the unlinked row would violate ``ux_documents_active_slot`` (23505)."""
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    channel = await seed_channel(sessions)
    assert channel.id is not None

    async with sessions.begin() as session:
        documents = SQLAlchemyDocumentRepository(session)
        saved = await documents.save(bound(channel.id, "docs/a.md"))
        await documents.save(
            Document(filename="docs/a.md", space_id=DEFAULT_SPACE_ID, status=DocumentStatus.DONE)
        )

    factory = session_factory(sessions)
    async with factory.create(master=True) as uow:
        document = await uow.documents.get_by_id(saved.id or 0)
        assert document is not None
        await unlink_document(uow, document, "uploads/public/1_a.md")

    async with sessions.begin() as session:
        unlinked = await SQLAlchemyDocumentRepository(session).get_by_id(saved.id or 0)
        assert unlinked is not None
        assert (unlinked.sync, unlinked.filename, unlinked.source_path, unlinked.ingestion_key) == (
            None,
            "docs/a(1).md",
            "uploads/public/1_a.md",
            None,
        )


async def test_storage_versions_latest_runs_and_tallies_on_postgres(pg_engine: AsyncEngine) -> None:
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    channel = await seed_channel(sessions)
    assert channel.id is not None

    async with sessions.begin() as session:
        documents = SQLAlchemyDocumentRepository(session)
        saved = await documents.save(bound(channel.id, "docs/a.md"))
        manual = await documents.save(Document(filename="m.md", space_id=DEFAULT_SPACE_ID))
        await documents.set_storage_version(saved.id or 0, "etag-2", "rev-2")
        # A manual document never takes a storage version.
        await documents.set_storage_version(manual.id or 0, "etag-x", "rev-x")
        runs = SQLAlchemySyncRunRepository(session)

        for _ in range(2):
            run = await runs.add(SyncRun(channel_id=channel.id, generation=1))
            run.start()
            run.tallies = {"unchanged": 3}
            run.complete(())
            await runs.update(run)

    async with sessions.begin() as session:
        documents = SQLAlchemyDocumentRepository(session)
        stored = await documents.get_by_id(saved.id or 0)
        assert stored is not None
        assert stored.sync is not None
        assert (stored.sync.storage_version, stored.sync.storage_revision) == ("etag-2", "rev-2")
        assert (await documents.get_by_id(manual.id or 0)).sync is None  # type: ignore[union-attr]
        latest = await SQLAlchemySyncRunRepository(session).latest_for_channels([channel.id, 999])
        assert list(latest) == [channel.id]
        assert latest[channel.id].id == max(
            run.id or 0 for run in await SQLAlchemySyncRunRepository(session).list_for_channel(channel.id)
        )
        assert latest[channel.id].counts() == {"unchanged": 3}
