"""
Composition Root — Dependency Injection Container (KinTree-style).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from application.services.auth_service import AuthService
from application.services.chat_service import ChatService
from application.services.config_service import ConfigService
from application.services.document_service import DocumentService
from application.services.ingest_service import IngestAppService
from application.uow import UnitOfWork
from config import settings
from domain.events.config_events import ConfigParameterChanged
from fastapi.security import APIKeyHeader
from infrastructure.auth.jwt_provider import JWTProvider
from infrastructure.auth.password_hasher import BCryptPasswordHasher
from infrastructure.database.database import database
from infrastructure.events.in_process_event_bus import event_bus
from infrastructure.events.postgres_config_broadcaster import PostgresConfigBroadcaster
from infrastructure.events.postgres_config_listener import PostgresConfigListener
from infrastructure.ml.config_subscribers import (
    apply_to_settings,
    audit_log_config_change,
    invalidate_bm25_cache_on_hybrid_toggle,
    invalidate_llm_cache,
    invalidate_paddle_ocr_cache,
    invalidate_storage_cache,
)
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser, LangchainDocumentSplitter
from infrastructure.ml.rag_service import RagService
from infrastructure.repositories.qdrant_vector_store_repository import QdrantVectorStoreRepository
from infrastructure.repositories.sqlalchemy_chunk_repository import SQLAlchemyChunkRepository
from infrastructure.services.benchmark_service import BenchmarkService
from infrastructure.services.ingestion_service import IngestionService
from infrastructure.storage import LazyStorage
from infrastructure.uow_factory import UnitOfWorkFactory

log = logging.getLogger("default")

# ---------------------------------------------------------------------------
# Event subscriptions (one-time at process start)
# ---------------------------------------------------------------------------

event_bus.subscribe(ConfigParameterChanged, apply_to_settings)
event_bus.subscribe(ConfigParameterChanged, invalidate_bm25_cache_on_hybrid_toggle)
event_bus.subscribe(ConfigParameterChanged, invalidate_llm_cache)
event_bus.subscribe(ConfigParameterChanged, invalidate_paddle_ocr_cache)
event_bus.subscribe(ConfigParameterChanged, invalidate_storage_cache)
event_bus.subscribe(ConfigParameterChanged, audit_log_config_change)

# ---------------------------------------------------------------------------
# Shared infrastructure instances (singletons)
# ---------------------------------------------------------------------------

_vector_store_repo = QdrantVectorStoreRepository()
_file_storage = LazyStorage()  # resolves lazily on first access; re-resolves after cache_clear()
_uow_factory = UnitOfWorkFactory(database=database)
_document_parser = LangchainDocumentParser()
_document_splitter = LangchainDocumentSplitter()
_config_broadcaster = PostgresConfigBroadcaster(database=database)


class _ChunkSearchAdapter:
    """Adapter that provides ChunkSearchPort using UoW factory for exact-search."""

    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def search_substring(
        self,
        query: str,
        user: dict,
        group_ids: list[int],
        assigned_client_ids: list[int],
        limit: int = 20,
        mode: str = "exact",
    ):
        async with self._uow_factory.create() as uow:
            repo = SQLAlchemyChunkRepository(uow._session)
            return await repo.search_substring(
                query=query,
                user=user,
                group_ids=group_ids,
                assigned_client_ids=assigned_client_ids,
                limit=limit,
                mode=mode,
            )


_chunk_search = _ChunkSearchAdapter(uow_factory=_uow_factory)

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
    rag_service=RagService(chunk_search=_chunk_search),
    history_window=settings.history_window,
)

_auth_service = AuthService(
    uow_factory=_uow_factory,
    password_hasher=BCryptPasswordHasher(),
    token_provider=JWTProvider(),
)

_config_service = ConfigService(
    uow_factory=_uow_factory, event_bus=event_bus, broadcaster=_config_broadcaster
)

_config_listener = PostgresConfigListener(event_bus, _uow_factory)

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
    return BenchmarkService()


def create_config_service() -> ConfigService:
    return _config_service


def get_config_listener() -> PostgresConfigListener:
    return _config_listener
