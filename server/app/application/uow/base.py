"""Base Unit of Work -- async transaction management with automatic commit/rollback.

Lives in the application layer so that infrastructure can inherit from it
without creating a reverse dependency.  Concrete subclasses (SQLAlchemy-backed)
live in ``infrastructure.database``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Self

from domain.shared.exceptions import DatabaseError


if TYPE_CHECKING:
    from types import TracebackType

    from application.uow.session import SessionProtocol


log = logging.getLogger("default")


class BaseUnitOfWork:
    """Base class for Unit of Work pattern.

    Manages a single database transaction across multiple repositories.
    Use as an async context manager to ensure commit on success,
    rollback on error, and session cleanup.
    """

    def __init__(self, session: SessionProtocol) -> None:
        self._session = session

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        try:
            if exc_type is None:
                await self._session.commit()
            else:
                await self.safe_rollback()
        except Exception as e:
            await self.safe_rollback()
            raise DatabaseError(detail=str(e)) from e
        finally:
            await self.safe_close()

    async def safe_rollback(self) -> None:
        """Roll back the transaction.

        A rollback failure is logged but does not mask the original error
        (the caller's exception is already in flight).
        """
        try:
            await self._session.rollback()
        except Exception:
            log.warning("Failed to roll back DB session cleanly", exc_info=True)

    async def safe_close(self) -> None:
        try:
            await self._session.close()
        except Exception:
            log.warning("Failed to close DB session cleanly", exc_info=True)

    async def publish_event(self, event: object) -> None:
        """Publish a domain event within the current transaction.

        Subclasses can override this to send notifications (e.g. pg_notify)
        atomically with the transaction commit.
        """
        del event
