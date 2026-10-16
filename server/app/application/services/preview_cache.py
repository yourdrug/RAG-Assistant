"""Compatibility exports; use application.documents."""

from application.documents.preview_cache import (
    DEFAULT_TTL_SECONDS,
    PREVIEW_PREFIX,
    PreviewCache,
)


__all__ = ["DEFAULT_TTL_SECONDS", "PREVIEW_PREFIX", "PreviewCache"]
