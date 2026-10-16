"""Compatibility exports; use application.documents."""

from application.documents.conflicts import (
    apply_async_replacement,
    resolve_conflict,
    resolve_conflict_in_uow,
    resolve_processing_conflict,
    resolve_upload_conflict,
    resolve_upload_version_group,
)


__all__ = [
    "apply_async_replacement",
    "resolve_conflict",
    "resolve_conflict_in_uow",
    "resolve_processing_conflict",
    "resolve_upload_conflict",
    "resolve_upload_version_group",
]
