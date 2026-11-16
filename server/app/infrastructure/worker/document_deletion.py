"""Arq adapter for durable document deletion."""

from __future__ import annotations

from typing import Any

from domain.runtime.jobs import JobType
from infrastructure.worker.runtime import job


@job(JobType.DOCUMENT_DELETION)
async def delete_document(ctx: dict[str, Any], *, document_id: int, user_id: int, job_id: int) -> None:  # noqa: ARG001 — consumed by @job
    await ctx["document_deletion_job"].run(document_id, user_id)
