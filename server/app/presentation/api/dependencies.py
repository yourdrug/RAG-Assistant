"""Composition Root — Dependency Injection Container.

Wires all use cases, repositories, and services together.
Every dependency flows inward: presentation → application → domain.
"""

from __future__ import annotations

import logging

from application.services.auth_service import AuthService
from application.services.chat_service import ChatService
from application.services.document_service import DocumentService
from application.services.ingest_service import IngestAppService
from application.use_cases.auth.authenticate_user import AuthenticateUser
from application.use_cases.auth.create_user import CreateUser
from application.use_cases.auth.list_users import ListUsers
from application.use_cases.auth.toggle_user_active import ToggleUserActive
from application.use_cases.chat.stream_chat import StreamChat
from application.use_cases.chat.sync_chat import SyncChat
from application.use_cases.document.delete_document import DeleteDocument
from application.use_cases.document.get_document import GetDocument
from application.use_cases.document.list_documents import ListDocuments
from application.use_cases.document.upload_document import UploadDocument
from application.use_cases.ingest.get_registry import GetIngestRegistry
from application.use_cases.ingest.ingest_single_file import IngestSingleFile
from application.use_cases.ingest.run_ingestion import RunIngestion
from config import settings
from infrastructure.auth.jwt_provider import JWTProvider
from infrastructure.auth.password_hasher import BCryptPasswordHasher
from infrastructure.ml.rag_service import RagService
from infrastructure.repositories.qdrant_vector_store_repository import QdrantVectorStoreRepository
from infrastructure.repositories.sqlalchemy_client_assignment_repository import (
    SQLAlchemyClientAssignmentRepository,
)
from infrastructure.repositories.sqlalchemy_conversation_repository import SQLAlchemyConversationRepository
from infrastructure.repositories.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
from infrastructure.repositories.sqlalchemy_group_repository import SQLAlchemyGroupRepository
from infrastructure.repositories.sqlalchemy_message_repository import SQLAlchemyMessageRepository
from infrastructure.repositories.sqlalchemy_user_repository import SQLAlchemyUserRepository
from infrastructure.services.benchmark_service import BenchmarkService
from infrastructure.services.document_processor import DocumentProcessor
from infrastructure.storage import get_storage
from sqlalchemy.orm import Session

log = logging.getLogger("default")


def get_repos(db: Session) -> dict:
    """Create all repository instances for a given DB session."""
    return {
        "user_repo": SQLAlchemyUserRepository(db),
        "conversation_repo": SQLAlchemyConversationRepository(db),
        "message_repo": SQLAlchemyMessageRepository(db),
        "document_repo": SQLAlchemyDocumentRepository(db),
        "group_repo": SQLAlchemyGroupRepository(db),
        "client_assignment_repo": SQLAlchemyClientAssignmentRepository(db),
    }


# ---------------------------------------------------------------------------
# Auth Service
# ---------------------------------------------------------------------------


def create_auth_service(db: Session) -> AuthService:
    repos = get_repos(db)
    hasher = BCryptPasswordHasher()
    token_provider = JWTProvider()

    return AuthService(
        authenticate_user=AuthenticateUser(
            user_repo=repos["user_repo"],
            password_verifier=hasher,
            token_provider=token_provider,
        ),
        create_user=CreateUser(
            user_repo=repos["user_repo"],
            password_hasher=hasher,
        ),
        list_users=ListUsers(user_repo=repos["user_repo"]),
        toggle_user_active=ToggleUserActive(user_repo=repos["user_repo"]),
    )


# ---------------------------------------------------------------------------
# Chat Service
# ---------------------------------------------------------------------------


def create_chat_service(db: Session) -> ChatService:
    repos = get_repos(db)
    rag_service = RagService()

    stream_chat = StreamChat(
        conversation_repo=repos["conversation_repo"],
        message_repo=repos["message_repo"],
        rag_service=rag_service,
        settings=settings,
    )
    sync_chat = SyncChat(
        conversation_repo=repos["conversation_repo"],
        message_repo=repos["message_repo"],
        rag_service=rag_service,
        settings=settings,
    )

    return ChatService(
        stream_chat=stream_chat,
        sync_chat=sync_chat,
    )


# ---------------------------------------------------------------------------
# Document Service
# ---------------------------------------------------------------------------


def create_document_service(db: Session) -> DocumentService:
    repos = get_repos(db)

    return DocumentService(
        upload_document=UploadDocument(
            document_repo=repos["document_repo"],
            group_repo=repos["group_repo"],
            document_processor=DocumentProcessor(db, repos["document_repo"]),
            file_storage=get_storage(),
        ),
        list_documents=ListDocuments(
            document_repo=repos["document_repo"],
            group_repo=repos["group_repo"],
            client_assignment_repo=repos["client_assignment_repo"],
        ),
        get_document=GetDocument(
            document_repo=repos["document_repo"],
            group_repo=repos["group_repo"],
            client_assignment_repo=repos["client_assignment_repo"],
        ),
        delete_document=DeleteDocument(
            document_repo=repos["document_repo"],
            vector_store_repo=QdrantVectorStoreRepository(),
            file_storage=get_storage(),
        ),
    )


# ---------------------------------------------------------------------------
# Ingest Service
# ---------------------------------------------------------------------------


def create_ingest_service() -> IngestAppService:
    from infrastructure.registry import load_registry, save_registry
    from infrastructure.services.ingestion_service import IngestionService

    ingestion = IngestionService()

    class PathResolver:
        def resolve_docs_dir(self, docs_dir: str) -> str:
            return ingestion.resolve_docs_dir(docs_dir)

        def resolve_ingest_target(self, file_path: str) -> str:
            return ingestion.resolve_ingest_target(file_path)

        def force_reindex(self, filename: str) -> None:
            ingestion.force_reindex(filename)

    class RegistryAdapter:
        def load(self) -> dict:
            return load_registry(settings.data_dir)

        def save(self, registry: dict) -> None:
            save_registry(settings.data_dir, registry)

    class IngestionAdapter:
        def run_full_ingestion(self, docs_dir: str, reset: bool = False, prefix: str | None = None) -> None:
            ingestion.run_full_ingestion(docs_dir, reset=reset, prefix=prefix)

        def run_single_file(self, file_path: str) -> None:
            ingestion.run_single_file(file_path)

    return IngestAppService(
        run_ingestion=RunIngestion(
            registry_repo=RegistryAdapter(),
            ingestion_service=IngestionAdapter(),
        ),
        ingest_single_file=IngestSingleFile(
            registry_repo=RegistryAdapter(),
            ingestion_service=IngestionAdapter(),
        ),
        get_registry=GetIngestRegistry(registry_repo=RegistryAdapter()),
        path_resolver=PathResolver(),
    )


# ---------------------------------------------------------------------------
# Ingestion Service (for upload endpoint)
# ---------------------------------------------------------------------------


def create_ingestion_service():
    from infrastructure.services.ingestion_service import IngestionService

    return IngestionService()


# ---------------------------------------------------------------------------
# Benchmark Service
# ---------------------------------------------------------------------------


def create_benchmark_service():
    from application.use_cases.benchmark.run_benchmark import RunBenchmark

    return RunBenchmark(benchmark_service=BenchmarkService(), settings=settings)
