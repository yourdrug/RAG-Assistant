"""Compatibility exports; use application.documents."""

from application.documents.utils import (
    generate_storage_key,
    resolve_unique_filename,
)


__all__ = ["generate_storage_key", "resolve_unique_filename"]
