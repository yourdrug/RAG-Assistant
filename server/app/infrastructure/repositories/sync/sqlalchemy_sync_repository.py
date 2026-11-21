"""SQLAlchemy persistence of sync channels and runs using the caller's session."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select

from domain.knowledge.enums import DocumentVisibility
from domain.sync.channel import (
    ChannelAudience,
    ChannelSettings,
    GitHubLocation,
    S3Location,
    StorageKind,
    SyncChannel,
)
from domain.sync.run import (
    TERMINAL_RUN_STATUSES,
    FileOutcome,
    FileOutcomeKind,
    SyncRun,
    SyncRunStatus,
    SyncTrigger,
)
from infrastructure.database.models import SyncChannelModel, SyncRunModel


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from domain.sync.channel import StorageLocation


def location_payload(location: StorageLocation) -> dict[str, Any]:
    if isinstance(location, S3Location):
        return {"bucket": location.bucket}

    return {
        "instance": location.instance,
        "repository_id": location.repository_id,
        "repository": location.repository,
        "branch": location.branch,
    }


def to_location(storage_kind: str, payload: dict[str, Any]) -> StorageLocation:
    if storage_kind == StorageKind.S3.value:
        return S3Location(**payload)

    return GitHubLocation(**payload)


def to_channel(orm: SyncChannelModel) -> SyncChannel:
    return SyncChannel(
        id=orm.id,
        space_id=orm.space_id,
        generation=orm.generation,
        creation_date=orm.creation_date,
        settings=ChannelSettings(
            location=to_location(orm.storage_kind, orm.location),
            audience=ChannelAudience(DocumentVisibility(orm.visibility), orm.owner_id, orm.group_id),
            responsible_user_id=orm.responsible_user_id,
            folder=orm.folder,
            exclusions=tuple(orm.exclusions or ()),
        ),
    )


def channel_columns(channel: SyncChannel) -> dict[str, Any]:
    settings = channel.settings
    return {
        "space_id": channel.space_id,
        "storage_kind": channel.storage_kind.value,
        "location": location_payload(settings.location),
        "folder": settings.folder,
        "exclusions": list(settings.exclusions),
        "visibility": settings.audience.visibility.value,
        "owner_id": settings.audience.owner_id,
        "group_id": settings.audience.group_id,
        "responsible_user_id": settings.responsible_user_id,
        "generation": channel.generation,
    }


class SQLAlchemySyncChannelRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_by_id(self, channel_id: int, *, for_update: bool = False) -> SyncChannel | None:
        stmt = select(SyncChannelModel).where(SyncChannelModel.id == channel_id)

        if for_update:
            stmt = stmt.with_for_update().execution_options(populate_existing=True)

        orm = (await self._db.execute(stmt)).scalar_one_or_none()
        return to_channel(orm) if orm is not None else None

    async def list_by_space(self, space_id: int) -> list[SyncChannel]:
        result = await self._db.execute(
            select(SyncChannelModel)
            .where(SyncChannelModel.space_id == space_id)
            .order_by(SyncChannelModel.id)
        )
        return [to_channel(orm) for orm in result.scalars().all()]

    async def list_all(self) -> list[SyncChannel]:
        result = await self._db.execute(select(SyncChannelModel).order_by(SyncChannelModel.id))
        return [to_channel(orm) for orm in result.scalars().all()]

    async def find_by_location(self, instance: str, repository_id: int, branch: str) -> list[SyncChannel]:
        result = await self._db.execute(
            select(SyncChannelModel)
            .where(
                func.lower(SyncChannelModel.location["instance"].astext) == instance.lower(),
                SyncChannelModel.location["repository_id"].astext == str(repository_id),
                SyncChannelModel.location["branch"].astext == branch,
            )
            .order_by(SyncChannelModel.id)
        )
        return [to_channel(orm) for orm in result.scalars().all()]

    async def add(self, channel: SyncChannel) -> SyncChannel:
        orm = SyncChannelModel(**channel_columns(channel), creation_date=channel.creation_date)
        self._db.add(orm)
        await self._db.flush()
        channel.id = orm.id
        return channel

    async def update(self, channel: SyncChannel) -> None:
        orm = await self._db.get(SyncChannelModel, channel.id)

        if orm is None:
            return

        for name, value in channel_columns(channel).items():
            setattr(orm, name, value)

        await self._db.flush()

    async def delete(self, channel_id: int) -> None:
        await self._db.execute(delete(SyncChannelModel).where(SyncChannelModel.id == channel_id))
        await self._db.flush()


def to_run(orm: SyncRunModel) -> SyncRun:
    return SyncRun(
        id=orm.id,
        channel_id=orm.channel_id,
        generation=orm.generation,
        trigger=SyncTrigger(orm.trigger),
        delivery_id=orm.delivery_id,
        status=SyncRunStatus(orm.status),
        storage_revision=orm.storage_revision,
        error=orm.error,
        outcomes=tuple(
            FileOutcome(item["path"], FileOutcomeKind(item["kind"]), item.get("detail"))
            for item in orm.outcomes
        ),
        creation_date=orm.creation_date,
        started_at=orm.started_at,
        finished_at=orm.finished_at,
    )


def run_columns(run: SyncRun) -> dict[str, Any]:
    return {
        "channel_id": run.channel_id,
        "generation": run.generation,
        "trigger": run.trigger.value,
        "delivery_id": run.delivery_id,
        "status": run.status.value,
        "storage_revision": run.storage_revision,
        "error": run.error,
        "outcomes": [
            {"path": item.path, "kind": item.kind.value, **({"detail": item.detail} if item.detail else {})}
            for item in run.outcomes
        ],
        "started_at": run.started_at,
        "finished_at": run.finished_at,
    }


class SQLAlchemySyncRunRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def add(self, run: SyncRun) -> SyncRun:
        orm = SyncRunModel(**run_columns(run), creation_date=run.creation_date)
        self._db.add(orm)
        await self._db.flush()
        run.id = orm.id
        return run

    async def update(self, run: SyncRun) -> None:
        orm = await self._db.get(SyncRunModel, run.id)

        if orm is None:
            return

        for name, value in run_columns(run).items():
            setattr(orm, name, value)

        await self._db.flush()

    async def get_by_id(self, run_id: int) -> SyncRun | None:
        orm = await self._db.get(SyncRunModel, run_id)
        return to_run(orm) if orm is not None else None

    async def list_unfinished(self, channel_id: int) -> list[SyncRun]:
        result = await self._db.execute(
            select(SyncRunModel)
            .where(
                SyncRunModel.channel_id == channel_id,
                SyncRunModel.status.not_in([status.value for status in TERMINAL_RUN_STATUSES]),
            )
            .order_by(SyncRunModel.id)
        )
        return [to_run(orm) for orm in result.scalars().all()]

    async def find_by_delivery(self, channel_id: int, delivery_id: str) -> SyncRun | None:
        result = await self._db.execute(
            select(SyncRunModel).where(
                SyncRunModel.channel_id == channel_id, SyncRunModel.delivery_id == delivery_id
            )
        )
        orm = result.scalar_one_or_none()
        return to_run(orm) if orm is not None else None

    async def list_for_channel(self, channel_id: int, limit: int = 20) -> list[SyncRun]:
        result = await self._db.execute(
            select(SyncRunModel)
            .where(SyncRunModel.channel_id == channel_id)
            .order_by(SyncRunModel.id.desc())
            .limit(limit)
        )
        return [to_run(orm) for orm in result.scalars().all()]
