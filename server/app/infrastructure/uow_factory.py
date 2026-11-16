"""Unit of Work factory -- creates async UnitOfWork instances bound to the DatabaseManager.

Each call to ``create()`` yields a fresh UnitOfWork with its own read/write
sessions.  The ``master=True`` variant uses the write session for mutations
that must bypass read replicas.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy import text

from application.jobs.execution import current_execution
from application.runtime.config import ConfigParameterChanged
from application.uow import UnitOfWork, UnitOfWorkRepositories
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_question_repository import (
    SQLAlchemyBenchmarkQuestionRepository,
)
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_run_repository import (
    SQLAlchemyBenchmarkRunRepository,
)
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import (
    SQLAlchemyChunkRepository,
)
from infrastructure.repositories.config.sqlalchemy_config_parameter_repository import (
    SQLAlchemyConfigParameterRepository,
)
from infrastructure.repositories.config.sqlalchemy_vector_outbox_repository import (
    SQLAlchemyVectorOutboxRepository,
)
from infrastructure.repositories.conversation.sqlalchemy_chat_log_repository import (
    SQLAlchemyChatLogRepository,
)
from infrastructure.repositories.conversation.sqlalchemy_conversation_repository import (
    SQLAlchemyConversationRepository,
)
from infrastructure.repositories.conversation.sqlalchemy_message_repository import (
    SQLAlchemyMessageRepository,
)
from infrastructure.repositories.document.sqlalchemy_document_repository import (
    SQLAlchemyDocumentRepository,
)
from infrastructure.repositories.document.sqlalchemy_ingestion_registry_repository import (
    SQLAlchemyIngestionRegistryRepository,
)
from infrastructure.repositories.misc.sqlalchemy_background_job_repository import (
    SQLAlchemyBackgroundJobRepository,
)
from infrastructure.repositories.space.sqlalchemy_space_repository import (
    SQLAlchemyKnowledgeSpaceRepository,
)
from infrastructure.repositories.user.sqlalchemy_api_key_repository import (
    SQLAlchemyApiKeyRepository,
)
from infrastructure.repositories.user.sqlalchemy_assignment_repository import (
    SQLAlchemyAssignmentRepository,
)
from infrastructure.repositories.user.sqlalchemy_group_repository import (
    SQLAlchemyGroupRepository,
)
from infrastructure.repositories.user.sqlalchemy_user_repository import (
    SQLAlchemyUserRepository,
)


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from infrastructure.database.database import DatabaseManager
    from infrastructure.domain_profile.packs import PackRepositoriesFactory
    from infrastructure.events.postgres_config_broadcaster import PostgresConfigBroadcaster


class UnitOfWorkFactory:
    """Factory for creating Unit of Work instances.

    Each call to create() yields a new UoW with a fresh async session.
    """

    def __init__(
        self,
        database: DatabaseManager,
        config_broadcaster: PostgresConfigBroadcaster | None = None,
        *,
        pack_repositories: Sequence[PackRepositoriesFactory] = (),
    ) -> None:
        self._database = database
        self._config_broadcaster = config_broadcaster
        # Build the loaded domain packs' repositories over each UoW session.
        self._pack_repositories = tuple(pack_repositories)

    @asynccontextmanager
    async def ingestion_lock(self) -> AsyncGenerator[None, None]:
        # NullPool closes the physical session even if acquiring or releasing the lock fails.
        async with self._database.async_master_connection() as connection:
            await connection.execute(text("SELECT pg_advisory_lock(724193820146)"))
            try:
                yield
            finally:
                await connection.execute(text("SELECT pg_advisory_unlock(724193820146)"))

    @asynccontextmanager
    async def document_index_lock(self, document_id: int) -> AsyncGenerator[None, None]:
        # Session lock on a nonpooled connection; no transaction spans vector I/O.
        async with self._database.async_master_connection() as connection:
            parameters = {"namespace": 724193, "document_id": document_id}
            await connection.execute(text("SELECT pg_advisory_lock(:namespace, :document_id)"), parameters)
            try:
                yield
            finally:
                await connection.execute(
                    text("SELECT pg_advisory_unlock(:namespace, :document_id)"), parameters
                )

    @asynccontextmanager
    async def create(self, *, master: bool = False) -> AsyncGenerator[UnitOfWork, None]:
        session = self._database.get_session(master=master)
        uow = UnitOfWork(
            session=session,
            repositories=UnitOfWorkRepositories(
                users=SQLAlchemyUserRepository(session),
                conversations=SQLAlchemyConversationRepository(session),
                messages=SQLAlchemyMessageRepository(session),
                documents=SQLAlchemyDocumentRepository(session),
                chunks=SQLAlchemyChunkRepository(session),
                groups=SQLAlchemyGroupRepository(session),
                api_keys=SQLAlchemyApiKeyRepository(session),
                config_parameters=SQLAlchemyConfigParameterRepository(session),
                background_jobs=SQLAlchemyBackgroundJobRepository(session),
                chat_logs=SQLAlchemyChatLogRepository(session),
                benchmark_questions=SQLAlchemyBenchmarkQuestionRepository(session),
                benchmark_sweeps=SQLAlchemyBenchmarkSweepRepository(session),
                benchmark_runs=SQLAlchemyBenchmarkRunRepository(session),
                vector_outbox=SQLAlchemyVectorOutboxRepository(session),
                assignments=SQLAlchemyAssignmentRepository(session),
                ingestion_registry=SQLAlchemyIngestionRegistryRepository(session),
                knowledge_spaces=SQLAlchemyKnowledgeSpaceRepository(session),
                pack_repositories=tuple(factory(session) for factory in self._pack_repositories),
            ),
        )

        if self._config_broadcaster is not None:
            broadcaster = self._config_broadcaster

            async def publish_to_broadcaster(event: object) -> None:
                if isinstance(event, ConfigParameterChanged):
                    await broadcaster.broadcast_within_session(session, event)

            uow.on_event(publish_to_broadcaster)

        async with uow:
            execution = current_execution.get() if master else None

            if execution is not None:
                await uow.background_jobs.lock_execution(execution.job_id, execution.attempt_id)

            yield uow

            if execution is not None:
                await uow.background_jobs.lock_execution(execution.job_id, execution.attempt_id)
