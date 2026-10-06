"""Infrastructure adapter for JobEnqueuerPort — wraps worker.queue functions."""

from __future__ import annotations

from infrastructure.worker.queue import (
    enqueue_benchmark as _enqueue_benchmark,
    enqueue_document_processing as _enqueue_document_processing,
    enqueue_ingest as _enqueue_ingest,
    enqueue_ingest_file as _enqueue_ingest_file,
    enqueue_sweep as _enqueue_sweep,
)


class JobEnqueuerAdapter:
    """Thin wrapper making worker queue functions available as an injectable port."""

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
    ) -> None:
        await _enqueue_document_processing(
            document_id=document_id,
            storage_key=storage_key,
            filename=filename,
            visibility=visibility,
            owner_id=owner_id,
            group_id=group_id,
            replace_id=replace_id,
            job_id=job_id,
            doc_domain=doc_domain,
            principal_id=principal_id,
        )

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
    ) -> None:
        await _enqueue_ingest(
            resolved_dir=resolved_dir,
            reset=reset,
            domain=domain,
            job_id=job_id,
            visibility=visibility,
            group_id=group_id,
            client_id=client_id,
        )

    async def enqueue_ingest_file(
        self,
        *,
        resolved: str,
        domain: str,
        job_id: int,
        visibility: str = "internal_public",
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> None:
        await _enqueue_ingest_file(
            resolved=resolved,
            domain=domain,
            job_id=job_id,
            visibility=visibility,
            group_id=group_id,
            client_id=client_id,
        )

    async def enqueue_benchmark(
        self,
        *,
        dataset: str,
        out_dir: str,
        top_k: int,
        judge_model: str,
        job_id: int,
    ) -> None:
        await _enqueue_benchmark(
            dataset=dataset,
            out_dir=out_dir,
            top_k=top_k,
            judge_model=judge_model,
            job_id=job_id,
        )

    async def enqueue_sweep(
        self,
        *,
        sweep_id: int,
        job_id: int,
    ) -> None:
        await _enqueue_sweep(sweep_id=sweep_id, job_id=job_id)
