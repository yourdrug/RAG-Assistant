"""main.py — Composition root for the RAG API."""

import logging.config
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from api.routes.auth import router as auth_router
from api.routes.chat import router as chat_router
from api.routes.clients import router as clients_router
from api.routes.conversations import router as conversations_router
from api.routes.documents import router as documents_router
from api.routes.groups import router as groups_router
from api.routes.health import router as health_router
from api.routes.ingest import router as ingest_router
from bootstrap import bootstrap_admin
from cli.cli import cli
from config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from infrastructure.logging import logging_config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logging.config.dictConfig(logging_config)
    bootstrap_admin()
    yield


def create_application() -> FastAPI:
    application = FastAPI(
        title="RAG API",
        description="Corporate RAG assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(auth_router)
    application.include_router(conversations_router)
    application.include_router(chat_router)
    application.include_router(ingest_router)
    application.include_router(documents_router)
    application.include_router(groups_router)
    application.include_router(clients_router)
    application.include_router(health_router)

    return application


app = create_application()

if __name__ == "__main__":
    logging.config.dictConfig(logging_config)
    cli.execute_command()
