"""UnitOfWorkFactory port -- application-layer abstraction for creating UnitOfWork instances.

Concrete implementations (SQLAlchemy-backed) live in ``infrastructure.uow_factory``.
The ``master=True`` flag selects the write session for mutations that must
bypass read replicas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from application.uow.unit_of_work import UnitOfWork


@runtime_checkable
class UnitOfWorkFactory(Protocol):
    def create(self, *, master: bool = False) -> AbstractAsyncContextManager[UnitOfWork]: ...

    def ingestion_lock(self) -> AbstractAsyncContextManager[None]: ...

    def document_index_lock(self, document_id: int) -> AbstractAsyncContextManager[None]: ...
