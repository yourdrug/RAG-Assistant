"""ChatContext value object -- encapsulates user access context for RAG retrieval.

Bundles the user's identity, group memberships, and assigned client IDs so
the retriever can build ACL-filtered Qdrant queries without reaching into
the infrastructure layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from domain.value_objects.curator_scope import CuratorScope
from domain.value_objects.roles import UserRole
from domain.value_objects.user_context import UserContext


@dataclass(frozen=True)
class ChatContext:
    """User access context passed to RAG service.

    Encapsulates ACL parameters that the RAG pipeline needs to build
    Qdrant filters. The domain defines the shape; infrastructure builds
    the concrete filter.
    """

    user_id: int
    user_kind: str
    user_role: str = UserRole.USER
    user_group_ids: list[int] = field(default_factory=list)
    curator_scope: CuratorScope | None = None
    depth: str | None = None
    summary: str | None = None
    as_of_date: date | None = None  # temporal retrieval: None = current state

    def to_user_context(self) -> UserContext:
        scope = self.curator_scope
        return UserContext(
            user_id=self.user_id,
            user_kind=self.user_kind,
            user_role=self.user_role,
            group_ids=tuple(self.user_group_ids),
            managed_client_ids=scope.managed_client_ids if scope else (),
            managed_internal_ids=scope.managed_internal_ids if scope else (),
            managed_group_ids=scope.managed_group_ids if scope else (),
        )
