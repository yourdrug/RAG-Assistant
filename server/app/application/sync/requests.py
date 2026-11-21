"""Request sync runs: manually, from a webhook delivery or by periodic reconciliation.

One run per channel: a new request cancels the unfinished ones. The run row and its
worker job commit together, so a committed request is always executed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from domain.runtime.jobs import BackgroundJob, BackgroundJobStatus, JobAdmissionError, JobType
from domain.shared.exceptions import BusinessRuleViolationError, EntityNotFoundError
from domain.sync.run import SyncRun, SyncRunStatus, SyncTrigger


if TYPE_CHECKING:
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.identity.user_context import UserContext


log = logging.getLogger("default")

ORPHANED_RUN_ERROR = "the worker stopped before the run finished"


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
        only_if_idle: bool = False,
    ) -> SyncRun | None:
        """Queue a run and its worker job; cancel unfinished runs of the channel.

        A manual request needs an administrator or the responsible user. A webhook
        redelivery (same ``delivery_id``) returns ``None`` and changes nothing. A run
        still waiting for a worker at the current generation already reads the newest
        storage state, so a further request joins it instead of queueing another job.
        ``only_if_idle`` (periodic reconciliation) queues nothing while a run is live.
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

            live = await live_runs(uow, channel_id)

            if only_if_idle and live:
                return None

            waiting = next(
                (
                    run
                    for run, job in live
                    if run.status == SyncRunStatus.QUEUED
                    and run.generation == channel.generation
                    and job.status == BackgroundJobStatus.PENDING
                ),
                None,
            )

            if waiting is not None:
                return waiting

            for unfinished, job in live:
                unfinished.cancel()
                await uow.sync_runs.update(unfinished)

                if job.id is not None and job.status == BackgroundJobStatus.PENDING:
                    # The run will not execute: free its queue slot at once.
                    await uow.background_jobs.cancel(job.id)

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
        """Periodic reconciliation: queue a run for every channel without a live run.

        It catches up on webhooks that were lost or never configured. Whether a channel
        is idle is decided again under its lock on the primary, so a run requested
        meanwhile is never cancelled, and a channel deleted meanwhile is skipped.
        """
        async with self._uow_factory.create() as uow:
            channel_ids = [c.id for c in await uow.sync_channels.list_all() if c.id is not None]

        queued = 0

        for channel_id in channel_ids:
            try:
                if await self.request(channel_id, SyncTrigger.SCHEDULE, only_if_idle=True):
                    queued += 1
            except EntityNotFoundError:
                continue
            except JobAdmissionError:
                log.warning("Scheduled sync stopped: the job queue is full")
                break

        return queued

    async def request_for_push(self, instance: str, repository_id: int, branch: str, delivery_id: str) -> int:
        """Queue a run for every channel reading the pushed branch; returns how many were queued.

        The delivery says only which branch changed: each run reads the storage itself.
        A push to a branch no channel reads queues nothing.
        """
        async with self._uow_factory.create() as uow:
            channels = await uow.sync_channels.find_by_location(instance, repository_id, branch)

        queued = 0

        for channel in channels:
            if channel.id is None:
                continue

            try:
                if await self.request(channel.id, SyncTrigger.WEBHOOK, delivery_id=delivery_id):
                    queued += 1
            except EntityNotFoundError:
                continue

        log.info("Webhook delivery queued %d sync run(s)", queued)
        return queued


async def live_runs(uow: UnitOfWork, channel_id: int) -> list[tuple[SyncRun, BackgroundJob]]:
    """Unfinished runs whose worker job can still execute them, with that job.

    A run whose job ended without finishing it (a stopped or killed worker, an expired
    lease) can never finish: it is failed here so it no longer blocks the channel.
    The caller holds the channel lock, which every run status change takes first.
    """
    live = []

    for run in await uow.sync_runs.list_unfinished(channel_id):
        job = await uow.background_jobs.find_latest_for(JobType.SYNC_RUN, run.id or 0)

        if job is None or BackgroundJobStatus(job.status).is_terminal:
            run.fail(ORPHANED_RUN_ERROR)
            await uow.sync_runs.update(run)
            log.warning("Sync run %s failed: its worker job ended without finishing it", run.id)
            continue

        live.append((run, job))

    return live
