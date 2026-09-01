"""Application bootstrap -- creates the default admin user and loads dynamic config from DB.

Called once during FastAPI lifespan startup.  If the admin user already
exists, the step is a no-op.  Dynamic config parameters are loaded from the
``config_parameters`` table and applied to in-memory settings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from bootstrap import bootstrap_admin
from config import settings
from domain.entities.benchmark_question import BenchmarkQuestion
from domain.events.config_events import ConfigParameterChanged

from infrastructure.events.in_process_event_bus import event_bus

logger = logging.getLogger("default")


async def initialize_app(uow_factory) -> None:
    """Run all startup initialization: bootstrap admin + load config from DB."""
    await _bootstrap_admin(uow_factory)
    await _load_config_from_db(uow_factory)
    await _migrate_test_questions(uow_factory)


async def _bootstrap_admin(uow_factory) -> None:
    """Ensure default admin user exists."""
    try:
        await bootstrap_admin(uow_factory)
    except Exception as e:
        logger.warning("Failed to bootstrap admin: %s", e)


async def _load_config_from_db(uow_factory) -> None:
    """При старте — прогнать все сохранённые параметры через событийную шину.

    Единый путь применения конфига: и runtime-обновления, и startup идут
    via ConfigParameterChanged → EventBus → подписчики.
    """
    try:
        async with uow_factory.create(master=True) as uow:
            rows = await uow.config_parameters.get_all()
            for r in rows:
                event_bus.publish(
                    ConfigParameterChanged(
                        key=r.key,
                        old_value=None,
                        new_value=r.value,
                        value_type=r.value_type,
                    )
                )
            logger.info("Loaded %d config parameters via event bus", len(rows))
    except Exception as e:
        logger.warning("Failed to load config from DB: %s", e)


async def _migrate_test_questions(uow_factory) -> None:
    """One-time migration: load test_questions.json into benchmark_questions table.

    If the table already has questions, this is a no-op.
    """
    try:
        async with uow_factory.create() as uow:
            count = await uow.benchmark_questions.count()

        if count > 0:
            return

        questions_file = Path(settings.data_dir) / "test_questions.json"
        if not questions_file.exists():
            return

        data = json.loads(questions_file.read_text(encoding="utf-8"))
        if not data:
            return

        entities = [
            BenchmarkQuestion(
                question=q.get("question", ""),
                expected_answer=q.get("expected_answer"),
                source_hint=q.get("source_hint"),
                dataset="main",
            )
            for q in data
            if q.get("question")
        ]

        if entities:
            async with uow_factory.create(master=True) as uow:
                imported = await uow.benchmark_questions.bulk_create(entities)
            logger.info(
                "Migrated %d questions from test_questions.json to benchmark_questions table",
                imported,
            )
    except Exception as e:
        logger.warning("Failed to migrate test_questions.json: %s", e)
