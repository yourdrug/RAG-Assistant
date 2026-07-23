"""Base Unit of Work — transaction management pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from types import TracebackType

from domain.exceptions import DatabaseError

from shared.session import SessionProtocol


class BaseUnitOfWork(ABC):
    """Abstract base class for Unit of Work pattern.

    Manages a single database transaction across multiple repositories.
    Use as a context manager to ensure commit on success,
    rollback on error, and session cleanup.
    """

    def __init__(self, session: SessionProtocol) -> None:
        self._session = session

    @property
    def session(self) -> SessionProtocol:
        """Public access to the underlying DB session."""
        return self._session

    @abstractmethod
    def __enter__(self) -> BaseUnitOfWork:
        return self

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._commit_or_rollback(exc_type)

    def _commit_or_rollback(self, exc_type: type[BaseException] | None) -> None:
        try:
            if exc_type is None:
                self._session.commit()
            else:
                with suppress(Exception):
                    self._session.rollback()
        except Exception as e:
            with suppress(Exception):
                self._session.rollback()
            raise DatabaseError(detail=str(e)) from e
        finally:
            with suppress(Exception):
                self._session.close()
