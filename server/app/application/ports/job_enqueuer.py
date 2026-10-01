"""JobEnqueuerPort — enqueues background jobs via the worker queue."""

from __future__ import annotations

from typing import Protocol


class JobEnqueuerPort(Protocol):
    """Enqueues background processing jobs."""

    async def enqueue_document_processing(
        self,
        *,
        document_id: int,
        storage_key: str,
        filename: str,
        visibility: str,
        owner_id: int | None,
        group_id: int | None,
        replace_id: int | None,
        job_id: int,
        doc_domain: str | None = None,
        principal_id: int | None = None,
    ) -> None: ...

    async def enqueue_ingest(
        self,
        *,
        resolved_dir: str,
        reset: bool,
        domain: str,
        job_id: int,
        visibility: str = "internal_public",
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> None: ...

    async def enqueue_ingest_file(
        self,
        *,
        resolved: str,
        domain: str,
        job_id: int,
        visibility: str = "internal_public",
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> None: ...

    async def enqueue_benchmark(
        self,
        *,
        questions_path: str,
        out_dir: str,
        top_k: int,
        judge_model: str,
        job_id: int,
    ) -> None: ...

    async def enqueue_sweep(
        self,
        *,
        sweep_id: int,
        job_id: int,
    ) -> None: ...
