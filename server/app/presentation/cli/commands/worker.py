"""CLI-команда: запуск Arq worker для обработки фоновых задач."""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence

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

        functions: Sequence = [
            process_document,
            run_full_ingest,
            run_single_ingest,
            run_benchmark,
        ]

        if max_jobs is None:
            max_jobs = settings.worker_max_concurrent

        w = Worker(
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
        w.run()
    except ImportError:
        logger.error("arq package not installed. Run: pip install arq redis")
        sys.exit(1)
    except Exception as exc:
        logger.error("Worker startup failed", exc_info=exc)
        sys.exit(1)


async def _on_startup(ctx: dict) -> None:
    """Initialize database and infrastructure on worker startup."""
    from composition.container import Container
    from infrastructure.database.database import database

    await database.connect()
    logger.info("Worker: database connected")

    # Build DI container (same as API process)
    container = Container()
    container.init(database)
    ctx["container"] = container

    listener = container.infrastructure.config_listener
    assert listener is not None

    await listener.resync(trigger="worker_startup")
    logger.info("Worker: config synced from database")

    await listener.start()
    ctx["config_listener"] = listener
    logger.info("Worker: config listener started")


async def _on_shutdown(ctx: dict) -> None:
    """Cleanup on worker shutdown."""
    from infrastructure.database.database import database

    listener = ctx.get("config_listener")
    if listener:
        await listener.stop()
        logger.info("Worker: config listener stopped")

    await database.disconnect()
    logger.info("Worker: database disconnected")
