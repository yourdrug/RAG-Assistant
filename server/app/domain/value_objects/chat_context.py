"""ChatContext value object -- encapsulates user access context for RAG retrieval.

Bundles the user's identity, group memberships, and assigned client IDs so
the retriever can build ACL-filtered Qdrant queries without reaching into
the infrastructure layer.
"""

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
    depth: str | None = None
