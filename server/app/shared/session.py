"""Session Protocol — abstract database session interface.

This allows the application layer to depend on a protocol
rather than a concrete ORM implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SessionProtocol(Protocol):
    """Abstract session interface matching SQLAlchemy Session API."""

    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...
    def execute(self, statement, params=None): ...
