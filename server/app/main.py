"""main.py — Composition root for the RAG API (provider-observer style)."""

from __future__ import annotations

import logging.config
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from cli.cli import cli
from config import settings
from domain.exceptions import ClientException, ServerException
from fastapi import FastAPI
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from infrastructure.auth.api_key_provider import api_key_provider
from infrastructure.database.database import database
from infrastructure.initialization import initialize_app
from infrastructure.logging import logging_config
from infrastructure.logging.log_buffer import attach_log_buffer
from infrastructure.ml.metrics import collect_infra_metrics
from infrastructure.ml.metrics_middleware import add_metrics_middleware
from infrastructure.persistence.redis_client import redis_client
from infrastructure.scheduler import scheduler
from infrastructure.utils import Singleton
from presentation.api.dependencies import _uow_factory, get_config_listener
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
from presentation.api.routes.api_keys import router as api_keys_router
from presentation.api.routes.auth import router as auth_router
from presentation.api.routes.benchmark import router as benchmark_router
from presentation.api.routes.chat import router as chat_router
from presentation.api.routes.clients import router as clients_router
from presentation.api.routes.conversations import router as conversations_router
from presentation.api.routes.documents import router as documents_router
from presentation.api.routes.groups import router as groups_router
from presentation.api.routes.health import router as health_router
from presentation.api.routes.ingest import router as ingest_router
from presentation.api.routes.search import router as search_router

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logging.config.dictConfig(logging_config)

    # Attach in-memory log buffer for /admin/logs endpoint
    attach_log_buffer()

    # Initialize Redis (mandatory component)
    await redis_client.init()

    # Initialize rate limiter (backed by the same Redis connection)
    await FastAPILimiter.init(redis_client.async_redis)

    await database.connect()
    await initialize_app(_uow_factory)
    await get_config_listener().start()

    # Start Pub/Sub listener for API key revocation (Redis mode)
    await api_key_provider.start_pubsub_listener()

    # Startup scheduler (periodic jobs)
    await scheduler.startup()

    # Initial infra metrics snapshot
    await collect_infra_metrics()

    yield

    # Shutdown
    await get_config_listener().stop()
    await scheduler.shutdown()
    await database.disconnect()
    await FastAPILimiter.close()
    await redis_client.aclose()


# ---------------------------------------------------------------------------
# Application (Singleton)
# ---------------------------------------------------------------------------


@Singleton
class Application:
    """FastAPI application configurator with structured setup."""

    def __init__(self) -> None:
        self.app: FastAPI = FastAPI(
            title="RAG API",
            description="Corporate RAG assistant",
            version=settings.version,
            lifespan=lifespan,
            servers=[{"url": "./", "description": "Relative server"}],
        )

        self.configure_logging()
        self.add_exception_handlers()
        self.add_middlewares()
        self.add_routers()
        add_metrics_middleware(self.app)

    def configure_logging(self) -> None:
        logging.config.dictConfig(logging_config)

    def add_exception_handlers(self) -> None:
        self.app.add_exception_handler(ClientException, handle_client_exception)  # type: ignore[arg-type]
        self.app.add_exception_handler(ServerException, handle_server_exception)  # type: ignore[arg-type]
        self.app.add_exception_handler(HTTPException, handle_http_exception)  # type: ignore[arg-type]
        self.app.add_exception_handler(RequestValidationError, handle_validation_exception)  # type: ignore[arg-type]
        self.app.add_exception_handler(Exception, handle_unexpected_exception)  # type: ignore[arg-type]

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
            groups_router,
            clients_router,
            health_router,
            benchmark_router,
            api_keys_router,
            admin_config_router,
            admin_chat_logs_router,
            admin_jobs_router,
            admin_metrics_router,
            admin_logs_router,
        )
        for router in routers:
            self.app.include_router(router)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_application() -> FastAPI:
    application: Application = Application()
    return application.app


app = create_application()

if __name__ == "__main__":
    logging.config.dictConfig(logging_config)
    cli.execute_command()
