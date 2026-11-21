"""Persistence ports for sync channels and their runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from domain.sync.channel import SyncChannel
    from domain.sync.run import SyncRun


@runtime_checkable
class SyncChannelRepository(Protocol):
    """Callers serialize channel changes of a space by locking its row (overlap checks)."""

    async def get_by_id(self, channel_id: int, *, for_update: bool = False) -> SyncChannel | None: ...

    async def list_by_space(self, space_id: int) -> list[SyncChannel]: ...

    async def add(self, channel: SyncChannel) -> SyncChannel: ...

    async def update(self, channel: SyncChannel) -> None:
        """Store settings and generation; the caller holds the channel row lock."""
        ...

    async def delete(self, channel_id: int) -> None: ...


@runtime_checkable
class SyncRunRepository(Protocol):
    async def add(self, run: SyncRun) -> SyncRun: ...

    async def update(self, run: SyncRun) -> None: ...

    async def get_by_id(self, run_id: int) -> SyncRun | None: ...

    async def list_for_channel(self, channel_id: int, limit: int = 20) -> list[SyncRun]:
        """Newest first."""
        ...
