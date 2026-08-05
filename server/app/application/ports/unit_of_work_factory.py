"""UnitOfWorkFactory port — application-layer abstraction for UoW creation."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Protocol

from application.uow import UnitOfWork


class UnitOfWorkFactory(Protocol):
    @asynccontextmanager
    async def create(self, master: bool = False) -> AsyncGenerator[UnitOfWork, None]: ...
