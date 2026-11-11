"""Real SQL unit-of-work factories for isolated integration sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import settings
from infrastructure.database.database import DatabaseManager
from infrastructure.uow_factory import UnitOfWorkFactory


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class SessionDatabase(DatabaseManager):
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        super().__init__(settings)
        self.sessions = sessions

    def get_session(self, *, master: bool = False) -> AsyncSession:
        del master
        return self.sessions()


def session_factory(sessions: async_sessionmaker[AsyncSession]) -> UnitOfWorkFactory:
    return UnitOfWorkFactory(SessionDatabase(sessions))
