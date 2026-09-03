"""Reusable presentation helpers — response mapping, document upload orchestration.

Eliminates duplicated DTO-to-response and document-upload-and-enqueue patterns
that were previously copy-pasted across route handlers.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from domain.value_objects.document_status import DocumentStatus
from infrastructure.logging.actions import log_action

from presentation.api.constants import JobType

logger = logging.getLogger("default")


# ---------------------------------------------------------------------------
# Generic list-response builder
# ---------------------------------------------------------------------------


def build_list_response(
    items: list[Any],
    total: int,
    *,
    container_key: str,
) -> dict:
    """Return ``{container_key: [...], "total": total}`` as a plain dict.

    Works with any Pydantic response model that follows the
    ``{items_key: [...], "total": int}`` convention.
    """
    return {container_key: items, "total": total}


# ---------------------------------------------------------------------------
# Source filtering helper (strip internal metadata keys)
# ---------------------------------------------------------------------------


def filter_sources(sources: list | None, *, exclude_keys: frozenset[str]) -> list | None:
    """Remove internal metadata keys (e.g. ``_confidence``) from source dicts."""
    if not sources:
        return None
    return [s for s in sources if not any(k in s for k in exclude_keys)]


# ---------------------------------------------------------------------------
# Document upload + job enqueue orchestration
# ---------------------------------------------------------------------------


@runtime_checkable
class _UploadableService(Protocol):
    async def upload(
        self,
        filename: str,
        file_data: bytes,
        visibility: str,
        group_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        client_id: int | None = ...,
        rename_on_conflict: bool = ...,
        doc_domain: str | None = ...,
    ) -> Any: ...


@runtime_checkable
class _JobService(Protocol):
    async def create_job(self, job_type: str, *, related_id: int | None = None) -> int: ...


async def upload_and_enqueue(
    *,
    file_data: bytes,
    filename: str,
    visibility: str,
    group_id: int | None,
    client_id: int | None,
    user_id: int,
    user_kind: str,
    user_role: str,
    rename_on_conflict: bool,
    doc_domain: str | None,
    document_service: _UploadableService,
    job_service: _JobService,
    enqueue_fn: Any,
    action_name: str,
) -> dict[str, Any]:
    """Shared upload → job-create → enqueue logic used by multiple routes.

    Returns a dict with ``document_id``, ``filename``, and ``status``.
    """
    result = await document_service.upload(
        filename=filename,
        file_data=file_data,
        visibility=visibility,
        group_id=group_id,
        client_id=client_id,
        user_id=user_id,
        user_kind=user_kind,
        user_role=user_role,
        rename_on_conflict=rename_on_conflict,
        doc_domain=doc_domain,
    )

    log_action(action_name, user_id=user_id, details={"filename": filename, "visibility": visibility})

    job_id = await job_service.create_job(JobType.DOCUMENT_PROCESSING, related_id=result.id)

    await enqueue_fn(
        document_id=result.id,
        storage_key=result.storage_key or "",
        filename=result.filename,
        visibility=visibility,
        owner_id=result.owner_id,
        group_id=group_id,
        replace_id=result.replace_id,
        doc_domain=doc_domain,
        job_id=job_id,
    )

    return {
        "document_id": result.id,
        "filename": filename,
        "status": DocumentStatus.PROCESSING.value,
    }
