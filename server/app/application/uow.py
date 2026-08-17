"""Unit of Work -- aggregates all repository interfaces into a single transaction boundary.

Each ``UnitOfWork`` instance exposes typed repository attributes (documents,
users, conversations, etc.) and a ``commit()`` / ``rollback()`` lifecycle.
Created by the ``UnitOfWorkFactory`` port and consumed by application services.
"""

from __future__ import annotations

from domain.repositories import ApiKeyRepository
from domain.repositories.background_job_repository import BackgroundJobRepository
from domain.repositories.benchmark_question_repository import BenchmarkQuestionRepository
from domain.repositories.benchmark_run_repository import BenchmarkRunRepository
from domain.repositories.benchmark_sweep_repository import BenchmarkSweepRepository
from domain.repositories.chat_log_repository import ChatLogRepository
from domain.repositories.chunk_repository import ChunkRepository
from domain.repositories.client_assignment_repository import ClientAssignmentRepository
from domain.repositories.config_parameter_repository import ConfigParameterRepository
from domain.repositories.conversation_repository import ConversationRepository
from domain.repositories.document_repository import DocumentRepository
from domain.repositories.group_repository import GroupRepository
from domain.repositories.message_repository import MessageRepository
from domain.repositories.user_repository import UserRepository
from infrastructure.database.base_uow import BaseUnitOfWork
from infrastructure.database.session_protocol import SessionProtocol


class UnitOfWork(BaseUnitOfWork):
    """Concrete UoW holding all application repositories.

    Usage:
        async with UnitOfWork(session) as uow:
            user = await uow.users.get_by_id(1)
            await uow.conversations.create(user_id=1)
            # Transaction commits automatically on clean exit
    """

    users: UserRepository
    conversations: ConversationRepository
    messages: MessageRepository
    documents: DocumentRepository
    chunks: ChunkRepository
    groups: GroupRepository
    client_assignments: ClientAssignmentRepository
    api_keys: ApiKeyRepository
    config_parameters: ConfigParameterRepository
    background_jobs: BackgroundJobRepository
    chat_logs: ChatLogRepository
    benchmark_questions: BenchmarkQuestionRepository
    benchmark_sweeps: BenchmarkSweepRepository
    benchmark_runs: BenchmarkRunRepository

    def __init__(
        self,
        session: SessionProtocol,
        users: UserRepository,
        conversations: ConversationRepository,
        messages: MessageRepository,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        groups: GroupRepository,
        client_assignments: ClientAssignmentRepository,
        api_keys: ApiKeyRepository,
        config_parameters: ConfigParameterRepository,
        background_jobs: BackgroundJobRepository,
        chat_logs: ChatLogRepository,
        benchmark_questions: BenchmarkQuestionRepository,
        benchmark_sweeps: BenchmarkSweepRepository,
        benchmark_runs: BenchmarkRunRepository,
    ) -> None:
        super().__init__(session)
        self.users = users
        self.conversations = conversations
        self.messages = messages
        self.documents = documents
        self.chunks = chunks
        self.groups = groups
        self.client_assignments = client_assignments
        self.api_keys = api_keys
        self.config_parameters = config_parameters
        self.background_jobs = background_jobs
        self.chat_logs = chat_logs
        self.benchmark_questions = benchmark_questions
        self.benchmark_sweeps = benchmark_sweeps
        self.benchmark_runs = benchmark_runs
