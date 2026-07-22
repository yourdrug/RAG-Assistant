"""main.py — Composition root for the RAG API."""

from __future__ import annotations

import logging.config
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from bootstrap import bootstrap_admin
from cli.cli import cli
from config import settings
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from infrastructure.logging import logging_config
from presentation.api.routes.auth import router as auth_router
from presentation.api.routes.benchmark import router as benchmark_router
from presentation.api.routes.chat import router as chat_router
from presentation.api.routes.clients import router as clients_router
from presentation.api.routes.conversations import router as conversations_router
from presentation.api.routes.documents import router as documents_router
from presentation.api.routes.groups import router as groups_router
from presentation.api.routes.health import router as health_router
from presentation.api.routes.ingest import router as ingest_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logging.config.dictConfig(logging_config)
    bootstrap_admin()
    yield


def create_application() -> FastAPI:
    application = FastAPI(
        title="RAG API",
        description="Corporate RAG assistant",
        version="0.2.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(ValidationError)
    async def validation_error_handler(_req: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @application.exception_handler(BusinessRuleViolation)
    async def business_rule_handler(_req: Request, exc: BusinessRuleViolation) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @application.exception_handler(EntityNotFound)
    async def not_found_handler(_req: Request, exc: EntityNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    application.include_router(auth_router)
    application.include_router(conversations_router)
    application.include_router(chat_router)
    application.include_router(ingest_router)
    application.include_router(documents_router)
    application.include_router(groups_router)
    application.include_router(clients_router)
    application.include_router(health_router)
    application.include_router(benchmark_router)

    return application


app = create_application()

if __name__ == "__main__":
    logging.config.dictConfig(logging_config)
    cli.execute_command()
