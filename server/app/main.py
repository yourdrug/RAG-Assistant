"""main.py — Composition root for the RAG API."""

from __future__ import annotations

import logging.config
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from bootstrap import bootstrap_admin
from cli.cli import cli
from config import settings
from domain.exceptions import DomainError
from fastapi import FastAPI
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from infrastructure.logging import logging_config
from presentation.api.exception_handlers import (
    handle_domain_exception,
    handle_http_exception,
    handle_unexpected_exception,
    handle_validation_exception,
)
from presentation.api.routes.auth import router as auth_router
from presentation.api.routes.benchmark import router as benchmark_router
from presentation.api.routes.chat import router as chat_router
from presentation.api.routes.clients import router as clients_router
from presentation.api.routes.conversations import router as conversations_router
from presentation.api.routes.documents import router as documents_router
from presentation.api.routes.groups import router as groups_router
from presentation.api.routes.health import router as health_router
from presentation.api.routes.ingest import router as ingest_router

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logging.config.dictConfig(logging_config)
    bootstrap_admin()
    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class Application:
    """FastAPI application configurator with structured setup."""

    def __init__(self) -> None:
        self.app = FastAPI(
            title="RAG API",
            description="Corporate RAG assistant",
            version="0.2.0",
            lifespan=lifespan,
        )

        self._configure_logging()
        self._add_middlewares()
        self._add_exception_handlers()
        self._add_routers()

    def _configure_logging(self) -> None:
        logging.config.dictConfig(logging_config)

    def _add_middlewares(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _add_exception_handlers(self) -> None:
        self.app.add_exception_handler(DomainError, handle_domain_exception)  # type: ignore[arg-type]
        self.app.add_exception_handler(HTTPException, handle_http_exception)  # type: ignore[arg-type]
        self.app.add_exception_handler(RequestValidationError, handle_validation_exception)  # type: ignore[arg-type]
        self.app.add_exception_handler(Exception, handle_unexpected_exception)  # type: ignore[arg-type]

    def _add_routers(self) -> None:
        self.app.include_router(auth_router)
        self.app.include_router(conversations_router)
        self.app.include_router(chat_router)
        self.app.include_router(ingest_router)
        self.app.include_router(documents_router)
        self.app.include_router(groups_router)
        self.app.include_router(clients_router)
        self.app.include_router(health_router)
        self.app.include_router(benchmark_router)


def create_application() -> FastAPI:
    """Create and return the configured FastAPI application."""
    application = Application()
    return application.app


app = create_application()

if __name__ == "__main__":
    logging.config.dictConfig(logging_config)
    cli.execute_command()
