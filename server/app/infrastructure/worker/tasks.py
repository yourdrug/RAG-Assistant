"""Arq task bodies; shared execution policies live in worker.runtime."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, Unpack

from application.documents.dto import ProcessDocumentCommand
from domain.knowledge.enums import DocumentVisibility
from infrastructure.worker.runtime import job


if TYPE_CHECKING:
    from application.ports.job_enqueuer import DocumentProcessingRequest


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document processing tasks
# ---------------------------------------------------------------------------


@job(per_principal=True)
async def process_document(ctx: dict[str, Any], **request: Unpack[DocumentProcessingRequest]) -> None:
    """Process an uploaded document (parse → split → vectorize → store)."""
    processor = ctx["document_processor"]
    logger.info(
        "Worker: background upload started: %s (doc %d, job %d)",
        request["filename"],
        request["document_id"],
        request["job_id"],
    )
    await processor.process(
        ProcessDocumentCommand(
            document_id=request["document_id"],
            storage_key=request["storage_key"],
            original_filename=request["filename"],
            visibility=DocumentVisibility.parse(request["visibility"]),
            owner_id=request["owner_id"],
            group_id=request["group_id"],
            replace_id=request["replace_id"],
            doc_domain=request.get("doc_domain"),
        )
    )
    logger.info(
        "Worker: background upload completed: %s (doc %d, job %d)",
        request["filename"],
        request["document_id"],
        request["job_id"],
    )


class IngestScope(TypedDict):
    domain: str
    visibility: str
    group_id: int | None
    client_id: int | None
    reset: NotRequired[bool]
    force: NotRequired[bool]


class IngestRequest(TypedDict):
    domain: str
    job_id: int
    visibility: NotRequired[str]
    group_id: NotRequired[int | None]
    client_id: NotRequired[int | None]


class FullIngestRequest(IngestRequest):
    resolved_dir: str
    reset: bool


class SingleIngestRequest(IngestRequest):
    resolved: str
    force: NotRequired[bool]


async def ingest_document_scope(
    ctx: dict[str, Any], path: str, *, single: bool, **scope: Unpack[IngestScope]
) -> None:
    """Dispatch ingestion with the same service and validated visibility scope."""
    service = ctx["ingestion_service"]
    vis = DocumentVisibility.parse(scope["visibility"])

    if single:
        await service.run_single(
            path,
            force=scope.get("force", False),
            domain=scope["domain"],
            visibility=vis,
            group_id=scope["group_id"],
            client_id=scope["client_id"],
        )
    else:
        await service.run_full(
            path,
            reset=scope.get("reset", False),
            domain=scope["domain"],
            visibility=vis,
            group_id=scope["group_id"],
            client_id=scope["client_id"],
        )


@job()
async def run_full_ingest(ctx: dict[str, Any], **request: Unpack[FullIngestRequest]) -> None:
    await ingest_document_scope(
        ctx,
        request["resolved_dir"],
        single=False,
        reset=request["reset"],
        domain=request["domain"],
        visibility=request.get("visibility", DocumentVisibility.INTERNAL_PUBLIC.value),
        group_id=request.get("group_id"),
        client_id=request.get("client_id"),
    )


@job()
async def run_single_ingest(ctx: dict[str, Any], **request: Unpack[SingleIngestRequest]) -> None:
    await ingest_document_scope(
        ctx,
        request["resolved"],
        single=True,
        force=request.get("force", False),
        domain=request["domain"],
        visibility=request.get("visibility", DocumentVisibility.INTERNAL_PUBLIC.value),
        group_id=request.get("group_id"),
        client_id=request.get("client_id"),
    )


@job()
async def run_benchmark(
    ctx: dict[str, Any],
    *,
    dataset: str,
    out_dir: str | None = None,
    top_k: int,
    judge_model: str,
    job_id: int,
) -> None:
    del out_dir  # Retained for jobs serialized before report export moved to the service.
    await ctx["benchmark_job"].run(dataset=dataset, top_k=top_k, judge_model=judge_model, job_id=job_id)
