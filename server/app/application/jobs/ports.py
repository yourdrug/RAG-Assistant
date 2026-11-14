"""Broker cancellation capability used by durable job management."""

from typing import Protocol


class JobAbortPort(Protocol):
    async def abort_job(self, job_id: int, job_type: str, related_id: int | None) -> None: ...
