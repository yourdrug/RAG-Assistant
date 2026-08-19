"""UnitOfWorkFactory port -- application-layer abstraction for creating UnitOfWork instances.

Concrete implementations (SQLAlchemy-backed) live in ``infrastructure.uow_factory``.
The ``master=True`` flag selects the write session for mutations that must
bypass read replicas.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Protocol, runtime_checkable

from application.uow import UnitOfWork


@runtime_checkable
class UnitOfWorkFactory(Protocol):
    @asynccontextmanager
    async def create(self, master: bool = False) -> AsyncGenerator[UnitOfWork, None]: ...
