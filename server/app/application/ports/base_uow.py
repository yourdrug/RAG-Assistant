"""Base Unit of Work -- async transaction management with automatic commit/rollback.

Lives in the application layer so that infrastructure can inherit from it
without creating a reverse dependency.  Concrete subclasses (SQLAlchemy-backed)
live in ``infrastructure.database``.
"""

from __future__ import annotations

import logging
from abc import ABC
from types import TracebackType

from domain.exceptions import DatabaseError

from application.ports.session_protocol import SessionProtocol

log = logging.getLogger("default")


class BaseUnitOfWork(ABC):  # noqa: B024
    """Abstract base class for Unit of Work pattern.

    Manages a single database transaction across multiple repositories.
    Use as an async context manager to ensure commit on success,
    rollback on error, and session cleanup.
    """

    def __init__(self, session: SessionProtocol) -> None:
        self._session = session

    async def __aenter__(self) -> BaseUnitOfWork:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if exc_type is None:
                await self._session.commit()
            else:
                await self._safe_rollback()
        except Exception as e:
            await self._safe_rollback()
            raise DatabaseError(detail=str(e)) from e
        finally:
            await self._safe_close()

    async def _safe_rollback(self) -> None:
        """Roll back the transaction.

        A rollback failure is logged but does not mask the original error
        (the caller's exception is already in flight).
        """
        try:
            await self._session.rollback()
        except Exception as e:
            log.warning("Failed to roll back DB session cleanly: %s", e)

    async def _safe_close(self) -> None:
        try:
            await self._session.close()
        except Exception as e:
            log.warning("Failed to close DB session cleanly: %s", e)

    async def publish_event(self, event: object) -> None:  # noqa: B027
        """Publish a domain event within the current transaction.

        Subclasses can override this to send notifications (e.g. pg_notify)
        atomically with the transaction commit.
        """
        pass
