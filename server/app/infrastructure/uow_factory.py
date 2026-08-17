"""Unit of Work factory -- creates async UnitOfWork instances bound to the DatabaseManager.

Each call to ``create()`` yields a fresh UnitOfWork with its own read/write
sessions.  The ``master=True`` variant uses the write session for mutations
that must bypass read replicas.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from application.uow import UnitOfWork

from infrastructure.database.database import DatabaseManager
from infrastructure.repositories.sqlalchemy_api_key_repository import SQLAlchemyApiKeyRepository
from infrastructure.repositories.sqlalchemy_background_job_repository import SQLAlchemyBackgroundJobRepository
from infrastructure.repositories.sqlalchemy_benchmark_question_repository import (
    SQLAlchemyBenchmarkQuestionRepository,
)
from infrastructure.repositories.sqlalchemy_benchmark_run_repository import SQLAlchemyBenchmarkRunRepository
from infrastructure.repositories.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)
from infrastructure.repositories.sqlalchemy_chat_log_repository import SQLAlchemyChatLogRepository
from infrastructure.repositories.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository
from infrastructure.repositories.sqlalchemy_client_assignment_repository import (
    SQLAlchemyClientAssignmentRepository,
)
from infrastructure.repositories.sqlalchemy_config_parameter_repository import (
    SQLAlchemyConfigParameterRepository,
)
from infrastructure.repositories.sqlalchemy_conversation_repository import SQLAlchemyConversationRepository
from infrastructure.repositories.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
from infrastructure.repositories.sqlalchemy_group_repository import SQLAlchemyGroupRepository
from infrastructure.repositories.sqlalchemy_message_repository import SQLAlchemyMessageRepository
from infrastructure.repositories.sqlalchemy_user_repository import SQLAlchemyUserRepository


class UnitOfWorkFactory:
    """Factory for creating Unit of Work instances.

    Each call to create() yields a new UoW with a fresh async session.
    """

    def __init__(self, database: DatabaseManager) -> None:
        self._database = database

    @asynccontextmanager
    async def create(self, master: bool = False) -> AsyncGenerator[UnitOfWork, None]:
        session = self._database.get_session(master=master)
        uow = UnitOfWork(
            session=session,
            users=SQLAlchemyUserRepository(session),
            conversations=SQLAlchemyConversationRepository(session),
            messages=SQLAlchemyMessageRepository(session),
            documents=SQLAlchemyDocumentRepository(session),
            chunks=SQLAlchemyChunkRepository(session),
            groups=SQLAlchemyGroupRepository(session),
            client_assignments=SQLAlchemyClientAssignmentRepository(session),
            api_keys=SQLAlchemyApiKeyRepository(session),
            config_parameters=SQLAlchemyConfigParameterRepository(session),
            background_jobs=SQLAlchemyBackgroundJobRepository(session),
            chat_logs=SQLAlchemyChatLogRepository(session),
            benchmark_questions=SQLAlchemyBenchmarkQuestionRepository(session),
            benchmark_sweeps=SQLAlchemyBenchmarkSweepRepository(session),
            benchmark_runs=SQLAlchemyBenchmarkRunRepository(session),
        )
        async with uow:
            yield uow
