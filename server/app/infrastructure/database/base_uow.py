"""Base Unit of Work — async transaction management pattern (KinTree-style)."""

from __future__ import annotations

from abc import ABC
from contextlib import suppress
from types import TracebackType

from domain.exceptions import DatabaseError
from sqlalchemy.exc import DBAPIError

from infrastructure.database.session_protocol import SessionProtocol


class BaseUnitOfWork(ABC):
    """Abstract base class for Unit of Work pattern.

    Manages a single database transaction across multiple repositories.
    Use as an async context manager to ensure commit on success,
    rollback on error, and session cleanup.
    """

    def __init__(self, session: SessionProtocol) -> None:
        self._session = session

    @property
    def session(self) -> SessionProtocol:
        return self._session

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
                with suppress(Exception):
                    await self._session.rollback()
        except DBAPIError as e:
            with suppress(Exception):
                await self._session.rollback()
            raise DatabaseError(detail=str(e)) from e
        finally:
            with suppress(Exception):
                await self._session.close()
