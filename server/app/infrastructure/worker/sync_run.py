"""Arq adapters for sync channel runs and their periodic reconciliation."""

from __future__ import annotations

from typing import Any

from domain.runtime.jobs import JobType
from infrastructure.worker.runtime import job


@job(JobType.SYNC_RUN)
async def run_sync(ctx: dict[str, Any], *, run_id: int, job_id: int) -> None:  # noqa: ARG001 — consumed by @job
    await ctx["sync_runs"].execute(run_id)


async def cron_sync_channels(ctx: dict[str, Any]) -> None:
    """Catch up on lost or unconfigured webhooks: one run per idle channel."""
    await ctx["sync_requests"].request_scheduled()
