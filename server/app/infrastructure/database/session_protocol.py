"""Async Session Protocol -- abstract database session interface.

Defines the ``SessionProtocol`` that both real SQLAlchemy ``AsyncSession``
and test mock sessions must satisfy.  Used for type-checking without
importing the concrete SQLAlchemy implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SessionProtocol(Protocol):
    """Abstract session interface matching SQLAlchemy AsyncSession API."""

    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...
    async def close(self) -> None: ...
    async def flush(self) -> None: ...
    async def execute(self, statement, params=None): ...
