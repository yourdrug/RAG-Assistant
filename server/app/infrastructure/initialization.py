"""Application bootstrap -- creates the default admin user and loads dynamic config from DB.

Called once during FastAPI lifespan startup.  If the admin user already
exists, the step is a no-op.  Dynamic config parameters are loaded from
the ``config_parameters`` table and applied to in-memory settings.
"""

from __future__ import annotations

import logging

from bootstrap import bootstrap_admin
from domain.events.config_events import ConfigParameterChanged

from infrastructure.events.in_process_event_bus import event_bus

logger = logging.getLogger("default")


async def initialize_app(uow_factory) -> None:
    """Run all startup initialization: bootstrap admin + load config from DB."""
    await _bootstrap_admin(uow_factory)
    await _load_config_from_db(uow_factory)


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
