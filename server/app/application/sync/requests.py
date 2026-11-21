"""Request sync runs: manually, from a webhook delivery or by periodic reconciliation.

One run per channel: a new request cancels the unfinished ones. The run row and its
worker job commit together, so a committed request is always executed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from domain.runtime.jobs import BackgroundJob, JobType
from domain.shared.exceptions import BusinessRuleViolationError, EntityNotFoundError
from domain.sync.run import SyncRun, SyncTrigger


if TYPE_CHECKING:
    from application.uow import UnitOfWorkFactory
    from domain.identity.user_context import UserContext


log = logging.getLogger("default")


class SyncRunRequests:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def request(
        self,
        channel_id: int,
        trigger: SyncTrigger,
        actor: UserContext | None = None,
        *,
        delivery_id: str | None = None,
    ) -> SyncRun | None:
        """Queue a run and its worker job; cancel unfinished runs of the channel.

        A manual request needs an administrator or the responsible user. A webhook
        redelivery (same ``delivery_id``) returns ``None`` and changes nothing.
        """
        async with self._uow_factory.create(master=True) as uow:
            channel = await uow.sync_channels.get_by_id(channel_id, for_update=True)

            if channel is None:
                msg = "Sync channel"
                raise EntityNotFoundError(msg, channel_id)

            if actor is not None and not channel.can_manage(actor.user_id, is_admin=actor.is_admin):
                msg = "Only an administrator or the channel's responsible user can start a sync"
                raise BusinessRuleViolationError(msg)

            if delivery_id is not None and await uow.sync_runs.find_by_delivery(channel_id, delivery_id):
                return None

            for unfinished in await uow.sync_runs.list_unfinished(channel_id):
                unfinished.cancel()
                await uow.sync_runs.update(unfinished)

            run = await uow.sync_runs.add(
                SyncRun(
                    channel_id=channel_id,
                    generation=channel.generation,
                    trigger=trigger,
                    delivery_id=delivery_id,
                )
            )
            await uow.background_jobs.create(
                BackgroundJob(
                    job_type=JobType.SYNC_RUN,
                    related_id=run.id,
                    dispatch_payload={"run_id": run.id, "cost_principal": f"sync_channel:{channel_id}"},
                )
            )

        return run

    async def request_scheduled(self) -> int:
        """Periodic reconciliation: queue a run for every channel without an unfinished one.

        It catches up on webhooks that were lost or never configured.
        """
        async with self._uow_factory.create() as uow:
            channels = await uow.sync_channels.list_all()
            idle = [
                c.id for c in channels if c.id is not None and not await uow.sync_runs.list_unfinished(c.id)
            ]

        for channel_id in idle:
            await self.request(channel_id, SyncTrigger.SCHEDULE)

        return len(idle)

    async def request_for_push(self, instance: str, repository_id: int, branch: str, delivery_id: str) -> int:
        """Queue a run for every channel reading the pushed branch; returns how many were queued.

        The delivery says only which branch changed: each run reads the storage itself.
        A push to a branch no channel reads queues nothing.
        """
        async with self._uow_factory.create() as uow:
            channels = await uow.sync_channels.find_by_location(instance, repository_id, branch)

        queued = 0

        for channel in channels:
            if channel.id is not None and await self.request(
                channel.id, SyncTrigger.WEBHOOK, delivery_id=delivery_id
            ):
                queued += 1

        log.info("Webhook delivery queued %d sync run(s)", queued)
        return queued
