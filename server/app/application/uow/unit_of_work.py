"""Typed repository collection and application transaction boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from application.uow.base import BaseUnitOfWork


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from application.uow.chunks import ChunkRepository
    from application.uow.session import SessionProtocol
    from domain.benchmark.repositories import (
        BenchmarkQuestionRepository,
        BenchmarkRunRepository,
        BenchmarkSweepRepository,
    )
    from domain.chat.repositories import ChatLogRepository, ConversationRepository, MessageRepository
    from domain.identity.repositories import (
        ApiKeyRepository,
        AssignmentRepository,
        GroupRepository,
        UserRepository,
    )
    from domain.ingestion.repositories import IngestionRegistryRepository
    from domain.knowledge.index_sync import VectorOutboxRepository
    from domain.knowledge.repositories.acts import ActVersionRepository, RegulatoryActRepository
    from domain.knowledge.repositories.document import DocumentRepository
    from domain.runtime.repositories import BackgroundJobRepository, ConfigParameterRepository


@dataclass(frozen=True)
class UnitOfWorkRepositories:
    """Repositories sharing the transaction session."""

    users: UserRepository
    conversations: ConversationRepository
    messages: MessageRepository
    documents: DocumentRepository
    chunks: ChunkRepository
    groups: GroupRepository
    api_keys: ApiKeyRepository
    config_parameters: ConfigParameterRepository
    background_jobs: BackgroundJobRepository
    chat_logs: ChatLogRepository
    benchmark_questions: BenchmarkQuestionRepository
    benchmark_sweeps: BenchmarkSweepRepository
    benchmark_runs: BenchmarkRunRepository
    vector_outbox: VectorOutboxRepository
    regulatory_acts: RegulatoryActRepository
    act_versions: ActVersionRepository
    assignments: AssignmentRepository
    ingestion_registry: IngestionRegistryRepository


class UnitOfWork(BaseUnitOfWork):
    """Expose repositories and publish events within one transaction."""

    def __init__(self, session: SessionProtocol, repositories: UnitOfWorkRepositories) -> None:
        super().__init__(session)
        self.users = repositories.users
        self.conversations = repositories.conversations
        self.messages = repositories.messages
        self.documents = repositories.documents
        self.chunks = repositories.chunks
        self.groups = repositories.groups
        self.api_keys = repositories.api_keys
        self.config_parameters = repositories.config_parameters
        self.background_jobs = repositories.background_jobs
        self.chat_logs = repositories.chat_logs
        self.benchmark_questions = repositories.benchmark_questions
        self.benchmark_sweeps = repositories.benchmark_sweeps
        self.benchmark_runs = repositories.benchmark_runs
        self.vector_outbox = repositories.vector_outbox
        self.regulatory_acts = repositories.regulatory_acts
        self.act_versions = repositories.act_versions
        self.assignments = repositories.assignments
        self.ingestion_registry = repositories.ingestion_registry
        self._event_handlers: list[Callable[[object], Coroutine[Any, Any, None]]] = []

    def on_event(self, handler: Callable[[object], Coroutine[Any, Any, None]]) -> None:
        """Register an async handler to be called on publish_event()."""
        self._event_handlers.append(handler)

    async def publish_event(self, event: object) -> None:
        for handler in self._event_handlers:
            await handler(event)
