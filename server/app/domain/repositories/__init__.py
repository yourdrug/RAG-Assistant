"""Repository interfaces -- abstract ports for persistence and external storage."""

from domain.repositories.act_version_repository import ActVersionRepository
from domain.repositories.background_job_repository import BackgroundJobRepository
from domain.repositories.benchmark_question_repository import BenchmarkQuestionRepository
from domain.repositories.benchmark_run_repository import BenchmarkRunRepository
from domain.repositories.benchmark_sweep_repository import BenchmarkSweepRepository
from domain.repositories.chat_log_repository import ChatLogRepository
from domain.repositories.chunk_repository import (
    ChunkContextRepository,
    ChunkCorpusRepository,
    ChunkCrudRepository,
    ChunkRepository,
    ChunkRetrievalRepository,
    ChunkSearchRepository,
    ChunkSearchResult,
    ChunkStats,
    ChunkVersioningRepository,
)
from domain.repositories.config_parameter_repository import ConfigParameterRepository
from domain.repositories.conversation_repository import ConversationRepository
from domain.repositories.document_repository import (
    DocumentCrudRepository,
    DocumentMaintenanceRepository,
    DocumentRepository,
)
from domain.repositories.ingestion_registry_repository import IngestionRegistryRepository
from domain.repositories.message_repository import MessageRepository
from domain.repositories.regulatory_act_repository import RegulatoryActRepository
from domain.repositories.vector_outbox_repository import (
    OutboxDispatcher,
    OutboxInspector,
    OutboxProducer,
    VectorOutboxRepository,
)
from domain.repositories.vector_store_repository import VectorStoreRepository

__all__ = [
    "ActVersionRepository",
    "BackgroundJobRepository",
    "BenchmarkQuestionRepository",
    "BenchmarkRunRepository",
    "BenchmarkSweepRepository",
    "ChatLogRepository",
    "ChunkContextRepository",
    "ChunkCorpusRepository",
    "ChunkCrudRepository",
    "ChunkRepository",
    "ChunkRetrievalRepository",
    "ChunkSearchRepository",
    "ChunkSearchResult",
    "ChunkStats",
    "ChunkVersioningRepository",
    "ConfigParameterRepository",
    "ConversationRepository",
    "DocumentCrudRepository",
    "DocumentMaintenanceRepository",
    "DocumentRepository",
    "IngestionRegistryRepository",
    "MessageRepository",
    "OutboxDispatcher",
    "OutboxInspector",
    "OutboxProducer",
    "RegulatoryActRepository",
    "VectorOutboxRepository",
    "VectorStoreRepository",
]
