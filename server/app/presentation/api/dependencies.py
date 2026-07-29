"""
Composition Root — Dependency Injection Container (KinTree-style).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from application.services.auth_service import AuthService
from application.services.chat_service import ChatService
from application.services.document_service import DocumentService
from application.services.ingest_service import IngestAppService
from application.uow import UnitOfWork
from config import settings
from fastapi.security import APIKeyHeader
from infrastructure.auth.jwt_provider import JWTProvider
from infrastructure.auth.password_hasher import BCryptPasswordHasher
from infrastructure.database.database import database
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser, LangchainDocumentSplitter
from infrastructure.ml.rag_service import RagService
from infrastructure.repositories.qdrant_vector_store_repository import QdrantVectorStoreRepository
from infrastructure.services.benchmark_service import BenchmarkService
from infrastructure.services.ingestion_service import IngestionService
from infrastructure.storage import get_storage
from infrastructure.uow_factory import UnitOfWorkFactory

log = logging.getLogger("default")

# ---------------------------------------------------------------------------
# Shared infrastructure instances (singletons)
# ---------------------------------------------------------------------------

_vector_store_repo = QdrantVectorStoreRepository()
_file_storage = get_storage()
_uow_factory = UnitOfWorkFactory(database=database)
_document_parser = LangchainDocumentParser()
_document_splitter = LangchainDocumentSplitter()

_ingestion_service = IngestionService(
    vector_store_repo=_vector_store_repo,
    file_storage=_file_storage,
    uow_factory=_uow_factory,
)

_document_service = DocumentService(
    uow_factory=_uow_factory,
    vector_store_repo=_vector_store_repo,
    file_storage=_file_storage,
)

_chat_service = ChatService(
    uow_factory=_uow_factory,
    rag_service=RagService(),
    history_window=settings.history_window,
)

_auth_service = AuthService(
    uow_factory=_uow_factory,
    password_hasher=BCryptPasswordHasher(),
    token_provider=JWTProvider(),
)

auth_key_header = APIKeyHeader(
    name="Authorization",
    auto_error=False,
)


# ---------------------------------------------------------------------------
# Unit of Work
# ---------------------------------------------------------------------------


async def get_uow() -> AsyncGenerator[UnitOfWork, None]:
    async with _uow_factory.create() as uow:
        yield uow


def get_uow_factory() -> UnitOfWorkFactory:
    return _uow_factory


# ---------------------------------------------------------------------------
# Application Services — every route depends on these, nothing builds its own
# ---------------------------------------------------------------------------


def create_ingest_service() -> IngestAppService:
    return IngestAppService(uow_factory=_uow_factory, ingestion_service=_ingestion_service)


def create_ingestion_service() -> IngestionService:
    return _ingestion_service


def create_document_service() -> DocumentService:
    return _document_service


def get_document_parser() -> LangchainDocumentParser:
    return _document_parser


def get_document_splitter() -> LangchainDocumentSplitter:
    return _document_splitter


def get_vector_store_repo() -> QdrantVectorStoreRepository:
    return _vector_store_repo


def get_file_storage():
    return _file_storage


def create_chat_service() -> ChatService:
    return _chat_service


def create_auth_service() -> AuthService:
    return _auth_service


def create_benchmark_service():
    from application.use_cases.benchmark.run_benchmark import RunBenchmark

    return RunBenchmark(benchmark_service=BenchmarkService(), settings=settings)
