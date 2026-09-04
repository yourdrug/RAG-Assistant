"""main.py — Composition root for the RAG API."""

from __future__ import annotations

import logging
import logging.config
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter

from composition.container import Container
from config import settings
from domain.exceptions import ClientException, ServerException
from infrastructure.database.database import database
from infrastructure.initialization import initialize_app
from infrastructure.logging import logging_config
from infrastructure.logging.log_buffer import attach_log_buffer
from infrastructure.ml.metrics import collect_infra_metrics
from infrastructure.ml.metrics_middleware import add_metrics_middleware
from infrastructure.persistence.redis_client import redis_client
from infrastructure.scheduler import scheduler
from infrastructure.utils import Singleton
from presentation.api.exception_handlers import (
    handle_client_exception,
    handle_http_exception,
    handle_server_exception,
    handle_unexpected_exception,
    handle_validation_exception,
)
from presentation.api.middleware.metrics import MetricsMiddleware
from presentation.api.middleware.request_id import RequestIDMiddleware
from presentation.api.routes.admin_chat_logs import router as admin_chat_logs_router
from presentation.api.routes.admin_config import router as admin_config_router
from presentation.api.routes.admin_jobs import router as admin_jobs_router
from presentation.api.routes.admin_logs import router as admin_logs_router
from presentation.api.routes.admin_metrics import router as admin_metrics_router
from presentation.api.routes.admin_quality import router as admin_quality_router
from presentation.api.routes.api_keys import router as api_keys_router
from presentation.api.routes.auth import router as auth_router
from presentation.api.routes.benchmark import router as benchmark_router
from presentation.api.routes.benchmark_admin import router as benchmark_admin_router
from presentation.api.routes.chat import router as chat_router
from presentation.api.routes.chunks import router as chunks_router
from presentation.api.routes.conversations import router as conversations_router
from presentation.api.routes.documents import router as documents_router
from presentation.api.routes.groups import router as groups_router
from presentation.api.routes.health import router as health_router
from presentation.api.routes.ingest import router as ingest_router
from presentation.api.routes.search import router as search_router
from presentation.cli.cli import cli

logger = logging.getLogger("default")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logging.config.dictConfig(logging_config)

    attach_log_buffer()
    await redis_client.init()
    await FastAPILimiter.init(redis_client.async_redis)
    await database.connect()

    # --- Build DI container (single call) ---
    container = Container()
    container.init(database)
    app.state.container = container

    if container.infrastructure.uow_factory is None:
        raise RuntimeError("UnitOfWorkFactory failed to initialize")
    await initialize_app(container.infrastructure.uow_factory)
    if container.infrastructure.config_listener is None:
        raise RuntimeError("ConfigListener failed to initialize")
    if container.infrastructure.api_key_provider is None:
        raise RuntimeError("ApiKeyProvider failed to initialize")
    await container.infrastructure.config_listener.start()
    if container.infrastructure.outbox_listener is not None:
        await container.infrastructure.outbox_listener.start()

    # Ensure Qdrant collection exists
    if container.infrastructure.vector_store_repo is not None:
        try:
            await container.infrastructure.vector_store_repo.ensure_collection(
                vector_size=settings.embed_dim,
                reset=False,
            )
            logger.info("Qdrant collection ensured (dim=%d)", settings.embed_dim)
        except Exception as e:
            logger.warning("Failed to ensure Qdrant collection: %s", e)

    await scheduler.startup(
        uow_factory=container.infrastructure.uow_factory,
        config_listener=container.infrastructure.config_listener,
        ml_clients=container.infrastructure.ml_clients,
        outbox_dispatcher=container.infrastructure.outbox_dispatcher,
    )
    await collect_infra_metrics(ml_clients=container.infrastructure.ml_clients)

    yield

    # --- Shutdown (reverse order) ---
    if container.infrastructure.outbox_listener is not None:
        await container.infrastructure.outbox_listener.stop()

    if container.infrastructure.config_listener is not None:
        await container.infrastructure.config_listener.stop()

    if container.infrastructure.ml_clients is not None:
        await container.infrastructure.ml_clients.close()

    await container.dispose()
    await scheduler.shutdown()
    await database.disconnect()
    await FastAPILimiter.close()
    await redis_client.aclose()


# ---------------------------------------------------------------------------
# Application (Singleton)
# ---------------------------------------------------------------------------


@Singleton
class Application:
    def __init__(self) -> None:
        self.app: FastAPI = FastAPI(
            title="RAG API",
            description="Corporate RAG assistant",
            version=settings.version,
            lifespan=lifespan,
            servers=[{"url": "./", "description": "Relative server"}],
        )
        self.add_exception_handlers()
        self.add_middlewares()
        self.add_routers()
        add_metrics_middleware(self.app)

    def add_exception_handlers(self) -> None:
        self.app.add_exception_handler(ClientException, handle_client_exception)
        self.app.add_exception_handler(ServerException, handle_server_exception)
        self.app.add_exception_handler(HTTPException, handle_http_exception)
        self.app.add_exception_handler(RequestValidationError, handle_validation_exception)
        self.app.add_exception_handler(Exception, handle_unexpected_exception)

    def add_middlewares(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.add_middleware(RequestIDMiddleware)
        self.app.add_middleware(MetricsMiddleware)

    def add_routers(self) -> None:
        routers = (
            auth_router,
            conversations_router,
            chat_router,
            ingest_router,
            search_router,
            documents_router,
            chunks_router,
            groups_router,
            health_router,
            benchmark_router,
            benchmark_admin_router,
            api_keys_router,
            admin_config_router,
            admin_chat_logs_router,
            admin_jobs_router,
            admin_metrics_router,
            admin_quality_router,
            admin_logs_router,
        )
        for router in routers:
            self.app.include_router(router)


def create_application() -> FastAPI:
    application: Application = Application()
    return application.app


app = create_application()

if __name__ == "__main__":
    logging.config.dictConfig(logging_config)
    cli.execute_command()
