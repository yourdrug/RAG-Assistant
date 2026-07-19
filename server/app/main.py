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
from infrastructure.singleton import Singleton


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logging.config.dictConfig(logging_config)
    bootstrap_admin()
    yield


@Singleton
class Application:
    def __init__(self) -> None:
        self.app: FastAPI = FastAPI(
            title="RAG API",
            description="Корпоративный ассистент на основе RAG",
            version="0.1.0",
            lifespan=lifespan,
        )

        self.configure_logging()
        self.add_middlewares()
        self.add_routers()

    def configure_logging(self) -> None:
        logging.config.dictConfig(logging_config)

    def add_middlewares(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def add_routers(self) -> None:
        self.app.include_router(auth_router)
        self.app.include_router(conversations_router)
        self.app.include_router(chat_router)
        self.app.include_router(ingest_router)
        self.app.include_router(documents_router)
        self.app.include_router(groups_router)
        self.app.include_router(clients_router)
        self.app.include_router(health_router)


def create_application() -> FastAPI:
    application = Application()
    return application.app


app = create_application()

if __name__ == "__main__":
    logging.config.dictConfig(logging_config)
    cli.execute_command()
