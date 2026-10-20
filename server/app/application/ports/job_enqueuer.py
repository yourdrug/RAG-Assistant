"""JobEnqueuerPort — enqueues background jobs via the worker queue."""

from __future__ import annotations

from typing import Protocol, TypedDict, Unpack

from domain.value_objects.visibility import DocumentVisibility


class DocumentProcessingOptions(TypedDict, total=False):
    doc_domain: str | None
    principal_id: int | None


class DocumentProcessingRequest(DocumentProcessingOptions):
    document_id: int
    storage_key: str
    filename: str
    visibility: str
    owner_id: int | None
    group_id: int | None
    replace_id: int | None
    job_id: int


class JobEnqueuerPort(Protocol):
    """Enqueues background processing jobs."""

    async def enqueue_document_deletion(self, *, document_id: int, user_id: int, job_id: int) -> None: ...

    async def enqueue_document_processing(self, **kwargs: Unpack[DocumentProcessingRequest]) -> None: ...

    async def enqueue_ingest(
        self,
        *,
        resolved_dir: str,
        reset: bool,
        domain: str,
        job_id: int,
        visibility: str = DocumentVisibility.INTERNAL_PUBLIC.value,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> None: ...

    async def enqueue_ingest_file(
        self,
        *,
        resolved: str,
        domain: str,
        job_id: int,
        visibility: str = DocumentVisibility.INTERNAL_PUBLIC.value,
        group_id: int | None = None,
        client_id: int | None = None,
        force: bool = False,
    ) -> None: ...

    async def enqueue_benchmark(
        self,
        *,
        dataset: str,
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
