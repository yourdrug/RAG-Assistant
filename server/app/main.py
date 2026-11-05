"""main.py — Composition root for the RAG API."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable

    from infrastructure.database.database import DatabaseManager
    from infrastructure.redis.redis_client import RedisClient
    from infrastructure.scheduling.scheduler import Scheduler
    from presentation.api.runtime import ApiContainer

from contextlib import asynccontextmanager
import logging
import logging.config

from fastapi import FastAPI
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from composition.cli.cli import cli
from composition.container import Container
from config import settings
from domain.shared.exceptions import AppClientError, AppServerError
from infrastructure.database.database import database
from infrastructure.initialization import bootstrap_admin
from infrastructure.logging import logging_config
from infrastructure.logging.log_buffer import attach_log_buffer
from infrastructure.metrics.metrics import collect_infra_metrics
from infrastructure.redis.redis_client import redis_client
from infrastructure.resilience.circuit_breaker import init_breakers
from infrastructure.scheduling.scheduler import scheduler
from infrastructure.utils.utils import singleton
from infrastructure.worker.queue import close_arq_pool
from presentation.api.exception_handlers import (
    handle_client_exception,
    handle_http_exception,
    handle_server_exception,
    handle_unexpected_exception,
    handle_validation_exception,
)
from presentation.api.middleware.metrics import MetricsMiddleware
from presentation.api.middleware.prometheus_setup import add_metrics_middleware
from presentation.api.middleware.request_id import RequestIDMiddleware
from presentation.api.routes.admin_act_versions import router as admin_act_versions_router
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
from presentation.api.routes.curators import router as curators_router
from presentation.api.routes.documents import router as documents_router
from presentation.api.routes.groups import router as groups_router
from presentation.api.routes.health import router as health_router
from presentation.api.routes.ingest import router as ingest_router
from presentation.api.routes.search import router as search_router


logger = logging.getLogger("default")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


async def shutdown_resources(
    container: Container | None, scheduler: Scheduler, database: DatabaseManager, redis_client: RedisClient
) -> None:
    """Gracefully shut down all resources. Each step isolated to prevent cascading failures."""

    async def safe_shutdown(coro: Awaitable[None], label: str) -> None:
        try:
            await coro
        except Exception:
            logger.warning("Failed to %s", label, exc_info=True)

    # container.dispose() handles: config_listener.stop(), outbox_listener.stop(),
    # ML client cache cleanup.  Do NOT duplicate those calls here.
    await safe_shutdown(scheduler.shutdown(), "shutdown scheduler")

    if container is not None:
        await safe_shutdown(container.dispose(), "dispose container")

    await safe_shutdown(close_arq_pool(), "close arq pool")
    await safe_shutdown(database.disconnect(), "disconnect database")
    await safe_shutdown(redis_client.aclose(), "close redis")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    logging.config.dictConfig(logging_config)

    container = None
    try:
        attach_log_buffer()
        await redis_client.init()
        await database.connect()

        # --- Build DI container (single call) ---
        container = Container()
        container.init(database)
        api_container: ApiContainer = container
        app.state.container = api_container

        if container.infrastructure.db.uow_factory is None:
            msg = "UnitOfWorkFactory failed to initialize"
            raise RuntimeError(msg)

        await container.infrastructure.bootstrap_configuration(role="api")
        await bootstrap_admin(container.infrastructure.db.uow_factory)

        await start_background_services(container)

        yield
    finally:
        await shutdown_resources(container, scheduler, database, redis_client)
        app.state.container = None


async def start_background_services(container: Container) -> None:
    if container.infrastructure.events.config_listener is None:
        msg = "ConfigListener failed to initialize"
        raise RuntimeError(msg)

    if container.infrastructure.services.api_key_provider is None:
        msg = "ApiKeyProvider failed to initialize"
        raise RuntimeError(msg)

    await container.infrastructure.events.config_listener.start()

    if container.infrastructure.events.outbox_listener is not None:
        await container.infrastructure.events.outbox_listener.start()

    # --- Circuit breakers ---
    init_breakers(
        fail_max=settings.llm_breaker_fail_max,
        timeout_duration=settings.llm_breaker_timeout_duration,
    )

    # Ensure Qdrant collection exists
    if container.infrastructure.ml.vector_store_repo is not None:
        corpus_control = container.infrastructure.corpus_control

        if corpus_control is not None and not await corpus_control.is_ready():
            logger.warning(
                "Skipping Qdrant collection initialization: corpus generation is not ready; "
                "continuing API startup with corpus operations unavailable."
            )
        else:
            await container.infrastructure.ml.vector_store_repo.ensure_collection(
                vector_size=settings.embed_dim,
                reset=False,
            )
            logger.info("Qdrant collection ensured (dim=%d)", settings.embed_dim)

    await scheduler.startup(
        config_listener=container.infrastructure.events.config_listener,
        ml_clients=container.infrastructure.ml.ml_clients,
        outbox_dispatcher=container.infrastructure.events.outbox_dispatcher,
    )
    await collect_infra_metrics(ml_clients=container.infrastructure.ml.ml_clients)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


@singleton
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
        self.app.add_exception_handler(AppClientError, handle_client_exception)
        self.app.add_exception_handler(AppServerError, handle_server_exception)
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
            admin_act_versions_router,
            curators_router,
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
