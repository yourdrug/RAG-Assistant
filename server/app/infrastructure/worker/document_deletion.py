"""Arq adapter for durable document deletion."""

from __future__ import annotations

from typing import Any

from infrastructure.worker.runtime import job


@job()
async def delete_document(ctx: dict[str, Any], *, document_id: int, user_id: int, job_id: int) -> None:  # noqa: ARG001 — consumed by @job
    await ctx["document_deletion_job"].run(document_id, user_id)
