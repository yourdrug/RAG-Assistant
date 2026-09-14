"""CuratorScope value object -- immutable snapshot of curator permissions.

Loaded from the assignment repository in UserContext.build(), projected into
ChatContext for RAG retrieval. Frozen and hashable so it can be included
in cache keys (Phase C).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CuratorScope:
    """Snapshot of curator managed-scope at request time.

    None in ChatContext means the user is not a curator (or has no managed IDs).
    Tuples (not lists) for immutability and hashability.
    """

    managed_client_ids: tuple[int, ...] = ()
    managed_internal_ids: tuple[int, ...] = ()
    managed_group_ids: tuple[int, ...] = ()

    def is_empty(self) -> bool:
        return not (self.managed_client_ids or self.managed_internal_ids or self.managed_group_ids)
