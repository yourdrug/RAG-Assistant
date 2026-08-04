"""ChatContext — value object encapsulating user access context for RAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChatContext:
    """User access context passed to RAG service.

    Encapsulates ACL parameters that the RAG pipeline needs to build
    Qdrant filters. The domain defines the shape; infrastructure builds
    the concrete filter.
    """

    user_id: int
    user_kind: str
    user_group_ids: list[int] = field(default_factory=list)
    assigned_client_ids: list[int] = field(default_factory=list)
    depth: str | None = None
