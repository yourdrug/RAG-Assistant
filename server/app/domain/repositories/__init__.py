"""Repository interfaces -- abstract ports for persistence and external storage."""

from domain.repositories.api_key_repository import ApiKeyRepository
from domain.repositories.benchmark_question_repository import BenchmarkQuestionRepository
from domain.repositories.benchmark_run_repository import BenchmarkRunRepository
from domain.repositories.benchmark_settings_repository import BenchmarkSettingsProtocol
from domain.repositories.benchmark_sweep_repository import BenchmarkSweepRepository
from domain.repositories.client_assignment_repository import ClientAssignmentRepository
from domain.repositories.conversation_repository import ConversationRepository
from domain.repositories.document_repository import DocumentRepository
from domain.repositories.group_repository import GroupRepository
from domain.repositories.ingestion_registry_repository import IngestionRegistryRepository
from domain.repositories.message_repository import MessageRepository
from domain.repositories.user_repository import UserRepository
from domain.repositories.vector_store_repository import VectorStoreRepository

__all__ = [
    "UserRepository",
    "ConversationRepository",
    "MessageRepository",
    "DocumentRepository",
    "GroupRepository",
    "ClientAssignmentRepository",
    "VectorStoreRepository",
    "IngestionRegistryRepository",
    "BenchmarkSettingsProtocol",
    "ApiKeyRepository",
    "BenchmarkQuestionRepository",
    "BenchmarkSweepRepository",
    "BenchmarkRunRepository",
]
