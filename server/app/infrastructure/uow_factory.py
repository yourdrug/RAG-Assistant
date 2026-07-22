"""Unit of Work Factory — creates UoW instances with all repository implementations."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from application.uow import UnitOfWork

from infrastructure.database.engine import SessionLocal
from infrastructure.repositories.sqlalchemy_client_assignment_repository import (
    SQLAlchemyClientAssignmentRepository,
)
from infrastructure.repositories.sqlalchemy_conversation_repository import SQLAlchemyConversationRepository
from infrastructure.repositories.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
from infrastructure.repositories.sqlalchemy_group_repository import SQLAlchemyGroupRepository
from infrastructure.repositories.sqlalchemy_message_repository import SQLAlchemyMessageRepository
from infrastructure.repositories.sqlalchemy_user_repository import SQLAlchemyUserRepository


class UnitOfWorkFactory:
    """Factory for creating Unit of Work instances.

    Each call to create() yields a new UoW with a fresh session.
    The session is committed on clean exit, rolled back on error.
    """

    @contextmanager
    def create(self) -> Generator[UnitOfWork, None, None]:
        session = SessionLocal()
        uow = UnitOfWork(
            session=session,
            users=SQLAlchemyUserRepository(session),
            conversations=SQLAlchemyConversationRepository(session),
            messages=SQLAlchemyMessageRepository(session),
            documents=SQLAlchemyDocumentRepository(session),
            groups=SQLAlchemyGroupRepository(session),
            client_assignments=SQLAlchemyClientAssignmentRepository(session),
        )
        with uow:
            yield uow
