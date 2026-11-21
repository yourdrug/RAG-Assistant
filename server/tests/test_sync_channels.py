"""Sync channels: exclusion rules, scope changes, file ownership and storage-managed documents."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from application.documents.delete import DocumentDeleteCommand
from application.documents.reindex import ReindexOutcome, skip_reason
from application.identity import UserContextFactory
from application.ingestion import DocumentSyncService
from application.sync import SyncChannelService
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.document import Document, SyncBinding
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.knowledge.index_sync import OutboxOperation
from domain.shared.exceptions import BusinessRuleViolationError, ValidationError
from domain.sync.channel import (
    ChannelAudience,
    ChannelDeletionMode,
    ChannelSettings,
    GitHubLocation,
    SyncChannel,
)
from domain.sync.ignore import IgnoreRules, InvalidIgnoreRulesError
from domain.sync.ownership import channels_overlap, ensure_no_overlap
from domain.sync.run import FileOutcome, FileOutcomeKind, SyncRun, SyncRunStatus
from tests.fakes import DEFAULT_SPACE_ID, UPLOADERS_GROUP_ID, FakeUnitOfWorkFactory, admit_all_users
from tests.test_chunk_service import _make_service, _make_uow


if TYPE_CHECKING:
    from tests.fakes import FakeUnitOfWork


ADMIN = UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN)
RESPONSIBLE = UserContext(user_id=2, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)
OTHER = UserContext(user_id=3, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)
PUBLIC = ChannelAudience(DocumentVisibility.INTERNAL_PUBLIC)


def location(branch: str = "main", repository_id: int = 42) -> GitHubLocation:
    return GitHubLocation("github.com", repository_id, "acme/docs", branch)


def settings(folder: str = "docs", exclusions: tuple[str, ...] = (), **changes: object) -> ChannelSettings:
    base = ChannelSettings(
        location=location(), audience=PUBLIC, responsible_user_id=RESPONSIBLE.user_id, folder=folder
    )
    return replace(base, exclusions=exclusions, **changes)  # type: ignore[arg-type]


def channel(channel_id: int, folder: str = "docs", **changes: Any) -> SyncChannel:
    return SyncChannel(space_id=DEFAULT_SPACE_ID, settings=settings(folder, **changes), id=channel_id)


# --- exclusion rules ------------------------------------------------------------------------


def test_nested_ragignore_applies_only_inside_its_folder() -> None:
    rules = IgnoreRules(ragignore={"guides/internal": "*.md"})
    assert rules.is_excluded("guides/internal/secret.md")
    assert rules.is_excluded("guides/internal/deep/secret.md")
    assert not rules.is_excluded("guides/public/secret.md")
    assert not rules.is_excluded("secret.md")


def test_deeper_ragignore_overrides_shallower_one() -> None:
    rules = IgnoreRules(ragignore={"": "*.draft.md", "guides": "!keep.draft.md"})
    assert rules.is_excluded("notes.draft.md")
    assert rules.is_excluded("guides/other.draft.md")
    assert not rules.is_excluded("guides/keep.draft.md")


def test_ragignore_cannot_undo_a_channel_exclusion() -> None:
    rules = IgnoreRules(["drafts/", "*.tmp.md"], {"": "!drafts/a.md\n!x.tmp.md"})
    assert rules.is_excluded("drafts/a.md")
    assert rules.is_excluded("x.tmp.md")


def test_directory_rule_matches_directories_at_any_depth_only() -> None:
    rules = IgnoreRules(["build/"])
    assert rules.is_excluded("build/a.md")
    assert rules.is_excluded("src/build/a.md")
    assert not rules.is_excluded("build.md")
    assert not rules.is_excluded("builds/a.md")


def test_file_inside_excluded_directory_cannot_be_reincluded() -> None:
    rules = IgnoreRules(ragignore={"": "drafts/\n!drafts/keep.md"})
    assert rules.is_excluded("drafts/keep.md")


def test_ragignore_itself_is_never_a_document() -> None:
    rules = IgnoreRules()
    assert rules.is_excluded(".ragignore")
    assert rules.is_excluded("guides/.ragignore")
    assert not rules.is_excluded("guides/ragignore.md")


def test_unparsable_patterns_name_their_source() -> None:
    with pytest.raises(InvalidIgnoreRulesError, match=r"guides/\.ragignore"):
        IgnoreRules(ragignore={"guides": "!"})

    with pytest.raises(InvalidIgnoreRulesError, match="channel settings"):
        settings(exclusions=("!",))


def test_folder_cannot_leave_the_storage_root() -> None:
    with pytest.raises(ValidationError):
        settings(folder="docs/../../etc")

    assert settings(folder="/docs/api/").folder == "docs/api"


# --- scope changes ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "narrowed",
    [settings("docs/api"), settings(exclusions=("drafts/",)), settings("docs/api", ("drafts/",))],
    ids=["narrower-folder", "added-exclusion", "both"],
)
def test_responsible_user_narrows_the_channel(narrowed: ChannelSettings) -> None:
    target = channel(1)
    target.change_settings(narrowed, user_id=RESPONSIBLE.user_id, is_admin=False)
    assert (target.settings, target.generation) == (narrowed, 2)


@pytest.mark.parametrize(
    "widened,reason",
    [
        (settings(""), "folder"),
        (settings("other"), "folder"),
        (settings(location=location("release")), "storage location"),
        (settings(audience=ChannelAudience(DocumentVisibility.INTERNAL_GROUP, group_id=7)), "audience"),
        (settings(responsible_user_id=OTHER.user_id), "responsible"),
        (settings(exclusions=("!drafts/keep.md",)), "exclusions"),
    ],
)
def test_only_an_administrator_widens_the_channel(widened: ChannelSettings, reason: str) -> None:
    target = channel(1)

    with pytest.raises(BusinessRuleViolationError, match=reason):
        target.change_settings(widened, user_id=RESPONSIBLE.user_id, is_admin=False)

    assert target.generation == 1
    target.change_settings(widened, user_id=ADMIN.user_id, is_admin=True)
    assert (target.settings, target.generation) == (widened, 2)


def test_removing_an_exclusion_is_widening() -> None:
    target = channel(1, exclusions=("drafts/",))

    with pytest.raises(BusinessRuleViolationError, match="exclusions"):
        target.change_settings(settings(), user_id=RESPONSIBLE.user_id, is_admin=False)


def test_other_users_cannot_change_a_channel_and_no_op_keeps_generation() -> None:
    target = channel(1)

    with pytest.raises(BusinessRuleViolationError):
        target.change_settings(settings("docs/api"), user_id=OTHER.user_id, is_admin=False)

    target.change_settings(settings(), user_id=RESPONSIBLE.user_id, is_admin=False)
    assert target.generation == 1


@pytest.mark.parametrize(
    "visibility,owner_id,group_id",
    [
        (DocumentVisibility.INTERNAL_PUBLIC, 5, None),
        (DocumentVisibility.INTERNAL_GROUP, None, None),
        (DocumentVisibility.CLIENT_PRIVATE, None, None),
        (DocumentVisibility.INTERNAL_PRIVATE, 5, 7),
    ],
)
def test_audience_needs_exactly_the_owner_or_group_of_its_visibility(
    visibility: DocumentVisibility, owner_id: int | None, group_id: int | None
) -> None:
    with pytest.raises(ValidationError):
        ChannelAudience(visibility, owner_id, group_id)


# --- file ownership -------------------------------------------------------------------------


def test_nested_channel_requires_exclusion_in_the_outer_channel() -> None:
    outer, inner = channel(1, "docs"), channel(2, "docs/api")
    assert channels_overlap(outer, inner)
    assert channels_overlap(channel(1, "docs"), channel(2, "docs"))
    assert channels_overlap(channel(1, ""), inner)

    excluded = channel(1, "docs", exclusions=("api/",))
    assert not channels_overlap(excluded, inner)
    # A pattern that leaves some files of the inner folder is not enough.
    assert channels_overlap(channel(1, "docs", exclusions=("api/*.draft.md",)), inner)

    with pytest.raises(BusinessRuleViolationError, match="overlap channel"):
        ensure_no_overlap(inner, [outer])

    ensure_no_overlap(inner, [excluded])


@pytest.mark.parametrize(
    "other",
    [
        channel(2, "docs", location=location("release")),
        channel(2, "docs", location=location(repository_id=7)),
        replace(channel(2, "docs"), space_id=DEFAULT_SPACE_ID + 1),
        channel(2, "guides"),
    ],
    ids=["other-branch", "other-repository", "other-space", "sibling-folder"],
)
def test_channels_reading_different_files_do_not_overlap(other: SyncChannel) -> None:
    assert not channels_overlap(channel(1, "docs"), other)


# --- runs ---------------------------------------------------------------------------------


def test_run_with_a_failed_file_is_partial() -> None:
    run = SyncRun(channel_id=1, generation=3)
    run.start("abc123")
    run.complete(
        (
            FileOutcome("a.md", FileOutcomeKind.ADDED),
            FileOutcome("b.md", FileOutcomeKind.FAILED, "too large"),
        )
    )
    assert (run.status, run.storage_revision, run.counts()) == (
        SyncRunStatus.PARTIAL,
        "abc123",
        {"added": 1, "failed": 1},
    )

    with pytest.raises(BusinessRuleViolationError):
        run.fail("late failure")

    run.cancel()
    assert run.status == SyncRunStatus.PARTIAL


# --- channel service ----------------------------------------------------------------------


def channel_world() -> tuple[FakeUnitOfWorkFactory, FakeUnitOfWork]:
    factory = FakeUnitOfWorkFactory()
    uow = factory._uow
    admit_all_users(uow)
    uow.fake_users.add_user(ADMIN.user_id)
    uow.fake_users.add_user(RESPONSIBLE.user_id)
    uow.fake_users.add_user(20, kind="client")
    return factory, uow


@pytest.mark.asyncio
async def test_only_an_administrator_creates_channels_and_overlaps_are_refused() -> None:
    factory, uow = channel_world()
    service = SyncChannelService(factory)

    with pytest.raises(BusinessRuleViolationError, match="administrator"):
        await service.create(DEFAULT_SPACE_ID, settings(), RESPONSIBLE)

    created = await service.create(DEFAULT_SPACE_ID, settings(), ADMIN)
    assert created.id is not None

    with pytest.raises(BusinessRuleViolationError, match="overlap"):
        await service.create(DEFAULT_SPACE_ID, settings("docs/api"), ADMIN)

    await service.change(created.id, settings(exclusions=("api/",)), RESPONSIBLE)
    await service.create(DEFAULT_SPACE_ID, settings("docs/api"), ADMIN)
    assert len(uow.fake_sync_channels.channels) == 2

    # Re-including the nested folder in the outer channel would overlap again.
    with pytest.raises(BusinessRuleViolationError, match="overlap"):
        await service.change(created.id, settings(), ADMIN)


@pytest.mark.asyncio
async def test_channel_rights_are_checked_on_save() -> None:
    factory, uow = channel_world()
    service = SyncChannelService(factory)
    client_audience = ChannelAudience(DocumentVisibility.CLIENT_PRIVATE, owner_id=20)
    await service.create(DEFAULT_SPACE_ID, settings(audience=client_audience), ADMIN)

    uow.fake_users._users[20].is_active = False

    with pytest.raises(BusinessRuleViolationError, match="owner"):
        await service.create(DEFAULT_SPACE_ID, settings("guides", audience=client_audience), ADMIN)

    # An internal user cannot own a client channel.
    internal_owner = ChannelAudience(DocumentVisibility.CLIENT_PRIVATE, owner_id=RESPONSIBLE.user_id)
    with pytest.raises(BusinessRuleViolationError, match="owner"):
        await service.create(DEFAULT_SPACE_ID, settings("guides", audience=internal_owner), ADMIN)

    group = ChannelAudience(DocumentVisibility.INTERNAL_GROUP, group_id=UPLOADERS_GROUP_ID + 1)
    with pytest.raises(BusinessRuleViolationError, match="group"):
        await service.create(DEFAULT_SPACE_ID, settings("guides", audience=group), ADMIN)

    uow.fake_users._users[RESPONSIBLE.user_id].is_active = False
    with pytest.raises(BusinessRuleViolationError, match="responsible"):
        await service.create(DEFAULT_SPACE_ID, settings("guides"), ADMIN)


async def bound_document(uow: FakeUnitOfWork, channel_id: int, path: str = "docs/a.md") -> Document:
    return await uow.documents.save(
        Document(
            filename=path, status=DocumentStatus.DONE, sync=SyncBinding(channel_id, path, "blob1", "c0ffee")
        )
    )


@pytest.mark.asyncio
async def test_unlinked_channel_documents_become_manual_with_their_ids() -> None:
    factory, uow = channel_world()
    service = SyncChannelService(factory)
    created = await service.create(DEFAULT_SPACE_ID, settings(), ADMIN)
    assert created.id is not None
    document = await bound_document(uow, created.id)

    assert await service.delete(created.id, ChannelDeletionMode.UNLINK, ADMIN) == 1
    unlinked = await uow.documents.get_by_id(document.id or 0)
    assert unlinked is not None
    assert (unlinked.sync, unlinked.visibility) == (None, document.visibility)
    assert uow.fake_sync_channels.channels == {}


@pytest.mark.asyncio
async def test_deleting_a_channel_deletes_its_documents_by_default() -> None:
    factory, uow = channel_world()
    service = SyncChannelService(factory)
    created = await service.create(DEFAULT_SPACE_ID, settings(), ADMIN)
    assert created.id is not None
    document = await bound_document(uow, created.id)
    manual = await uow.documents.save(Document(filename="manual.md", status=DocumentStatus.DONE))

    with pytest.raises(BusinessRuleViolationError):
        await service.delete(created.id, ChannelDeletionMode.DELETE_DOCUMENTS, RESPONSIBLE)

    assert await service.delete(created.id, ChannelDeletionMode.DELETE_DOCUMENTS, ADMIN) == 1
    assert await uow.documents.get_by_id(document.id or 0) is None
    assert await uow.documents.get_by_id(manual.id or 0) is not None
    assert [
        entry.aggregate_id
        for entry in uow.fake_vector_outbox._entries.values()
        if entry.operation == OutboxOperation.DELETE_BY_DOCUMENT
    ] == [document.id]


# --- storage-managed documents --------------------------------------------------------------


@pytest.mark.asyncio
async def test_manual_chunk_edit_of_channel_document_is_refused_until_unlinked() -> None:
    doc = Document(
        id=1, filename="docs/a.md", owner_id=10, status=DocumentStatus.DONE, sync=SyncBinding(5, "a.md")
    )
    uow = _make_uow(doc=doc)
    service = _make_service(uow)
    content = "Ручная правка текста документа для проверки ограничения канала. " * 4

    with pytest.raises(BusinessRuleViolationError, match="sync channel 5"):
        await service.edit_chunk(1, 10, content, 1, UserRole.ADMIN)

    with pytest.raises(BusinessRuleViolationError, match="sync channel 5"):
        await service.add_chunk(1, content, 1, UserRole.ADMIN)

    doc.sync = None
    await service.edit_chunk(1, 10, content, 1, UserRole.ADMIN)


@pytest.mark.asyncio
async def test_channel_document_cannot_be_deleted_by_hand() -> None:
    factory, uow = channel_world()
    document = await bound_document(uow, 5)
    command = DocumentDeleteCommand(factory, file_storage=None, user_ctx_factory=UserContextFactory())  # type: ignore[arg-type]

    with pytest.raises(BusinessRuleViolationError, match="sync channel"):
        await command.execute(document.id or 0, ADMIN.user_id, UserRole.ADMIN.value)

    assert await uow.documents.get_by_id(document.id or 0) is not None


def test_reindex_skips_channel_documents() -> None:
    document = Document(id=1, status=DocumentStatus.DONE, source_path="x.md", sync=SyncBinding(5, "x.md"))
    assert skip_reason(document) == (ReindexOutcome.SKIPPED_SOURCE, "managed by a sync channel")


@pytest.mark.asyncio
async def test_cli_prefix_cleanup_and_ingestion_never_touch_channel_documents() -> None:
    factory, uow = channel_world()
    # A channel document whose key lies under an ingested prefix (an S3 channel, later).
    document = await uow.documents.save(
        Document(
            filename="docs/a.md",
            ingestion_key="docs/a.md",
            status=DocumentStatus.DONE,
            sync=SyncBinding(5, "docs/a.md"),
        )
    )
    sync = DocumentSyncService(factory)
    await sync.delete_missing_sources("docs/", set(), space_id=DEFAULT_SPACE_ID)
    assert await uow.documents.get_by_id(document.id or 0) is not None

    with pytest.raises(BusinessRuleViolationError, match="sync channel"):
        await sync.ensure_document_rows({"docs/a.md": {"source": "s3://bucket/docs/a.md"}})  # type: ignore[typeddict-item]


@pytest.mark.asyncio
async def test_manual_upload_slot_ignores_channel_documents_with_the_same_name() -> None:
    _factory, uow = channel_world()
    await bound_document(uow, 5, "a.md")
    assert await uow.documents.find_active_slot(None, "a.md", None, space_id=DEFAULT_SPACE_ID) is None
