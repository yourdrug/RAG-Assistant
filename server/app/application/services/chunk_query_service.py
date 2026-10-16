"""Compatibility exports; use application.chunks."""

from application.chunks.queries import (
    ChunkQueryService,
    validate_chunk_acl,
)


__all__ = ["ChunkQueryService", "validate_chunk_acl"]
