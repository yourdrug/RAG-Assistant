"""
main.py — Slim composition root for the RAG API.

Endpoints:
  POST /auth/login          — JWT login
  GET  /auth/me              — current user
  POST /auth/users            — [admin] create user
  GET  /auth/users             — [admin] list users
  PATCH /auth/users/{id}        — [admin] toggle user active
  POST /chat                — streaming SSE [user]
  POST /chat/sync           — synchronous response [user]
  POST /conversations       — new conversation [user]
  GET  /conversations/{id}  — conversation history [user]
  POST /ingest               — [admin] index documents
  GET  /health               — healthcheck (no auth)
"""

from __future__ import annotations

import logging
import logging.config
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from cli.cli import cli
from config import settings
from fastapi import FastAPI

logger = logging.getLogger("default")


def bootstrap_admin():
    from infrastructure.auth import hash_password
    from infrastructure.database import (
        SessionLocal,
        any_admin_exists,
        create_user,
    )

    db = SessionLocal()
    try:
        if any_admin_exists(db):
            return
        if not settings.admin_email or not settings.admin_password:
            logger.warning(
                "Нет ни одного admin, а ADMIN_EMAIL/ADMIN_PASSWORD не заданы — "
                "залогиниться будет некому. Задай их в server/.env и перезапусти."
            )
            return
        create_user(
            db,
            email=settings.admin_email,
            hashed_password=hash_password(settings.admin_password),
            role="admin",
        )
        logger.info("Создан admin: %s", settings.admin_email)
    finally:
        db.close()


def _preload_models() -> None:
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    embed_cached = any(cache_dir.glob(f"models--{settings.embed_model.replace('/', '--')}*"))
    rerank_cached = any(cache_dir.glob(f"models--{settings.rerank_model.replace('/', '--')}*"))

    if embed_cached and rerank_cached:
        logger.info("Модели уже в кэше — пропускаю предзагрузку")
        return

    if not embed_cached:
        logger.info("Предзагрузка эмбеддинг-модели %s ...", settings.embed_model)
        from infrastructure.vector_store import get_embeddings

        get_embeddings()
    if not rerank_cached:
        logger.info("Предзагрузка реранкера %s ...", settings.rerank_model)
        from infrastructure.vector_store import get_reranker

        get_reranker()
    logger.info("Модели загружены")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    from infrastructure.logging import logging_config

    logging.config.dictConfig(logging_config)
    bootstrap_admin()
    _preload_models()
    yield


def create_application() -> FastAPI:
    from api.routes import Application

    application = Application()
    application.app.router.lifespan_context = lifespan
    return application.app


app = create_application()

if __name__ == "__main__":
    cli.execute_command()
