"""CLI-команда: запуск Arq worker для обработки фоновых задач."""

from __future__ import annotations

import logging
import sys

from arq.connections import RedisSettings
from arq.worker import Worker
from config import settings
from infrastructure.worker.tasks import (
    process_document,
    run_benchmark,
    run_full_ingest,
    run_single_ingest,
)

logger = logging.getLogger("cli")


def worker(
    max_jobs: int | None = None,
    health_check_interval: int = 10,
) -> None:
    """Запустить Arq worker для обработки фоновых задач.

    Слушает очереди: document_processing, ingest, benchmark.
    Использует Redis как брокер.
    """
    try:
        redis_settings = RedisSettings.from_dsn(settings.redis_url)

        functions = [
            process_document,
            run_full_ingest,
            run_single_ingest,
            run_benchmark,
        ]

        if max_jobs is None:
            max_jobs = settings.worker_max_concurrent

        worker = Worker(
            functions=functions,
            redis_settings=redis_settings,
            max_jobs=max_jobs,
            health_check_interval=health_check_interval,
            queue_name="document_processing",
            on_startup=_on_startup,
            on_shutdown=_on_shutdown,
        )

        logger.info(
            "Arq worker starting — queues: document_processing, ingest, benchmark max_jobs=%d redis=%s",
            max_jobs,
            settings.redis_host,
        )
        worker.run()
    except ImportError:
        logger.error("arq package not installed. Run: pip install arq redis")
        sys.exit(1)
    except Exception as exc:
        logger.error("Worker startup failed", exc_info=exc)
        sys.exit(1)


async def _on_startup(ctx: dict) -> None:
    """Initialize database and infrastructure on worker startup."""
    from infrastructure.database.database import database  # nested to avoid circular import
    from presentation.api.dependencies import get_config_listener, get_uow_factory

    await database.connect()
    logger.info("Worker: database connected")

    # Sync dynamic config from DB BEFORE starting the listener (and before
    # processing any tasks). This ensures ocr_enabled, chunk_size, etc.
    # are current even if the config was changed while the worker was down.
    listener = get_config_listener()
    await listener.resync(trigger="worker_startup")
    logger.info("Worker: config synced from database")

    # Now start the listener for future changes
    await listener.start()
    ctx["config_listener"] = listener
    logger.info("Worker: config listener started")


async def _on_shutdown(ctx: dict) -> None:
    """Cleanup on worker shutdown."""
    from infrastructure.database.database import database  # nested to avoid circular import

    # Stop config listener
    listener = ctx.get("config_listener")
    if listener:
        await listener.stop()
        logger.info("Worker: config listener stopped")

    await database.disconnect()
    logger.info("Worker: database disconnected")
