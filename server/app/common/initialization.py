"""Initialization — bootstrap admin user + load dynamic config from DB."""

from __future__ import annotations

import logging

logger = logging.getLogger("default")


async def initialize_app(uow_factory) -> None:
    """Run all startup initialization: bootstrap admin + load config from DB."""
    await _bootstrap_admin(uow_factory)
    await _load_config_from_db(uow_factory)


async def _bootstrap_admin(uow_factory) -> None:
    """Ensure default admin user exists."""
    from bootstrap import bootstrap_admin

    try:
        await bootstrap_admin(uow_factory)
    except Exception as e:
        logger.warning("Failed to bootstrap admin: %s", e)


async def _load_config_from_db(uow_factory) -> None:
    """Load dynamic config from DB and apply to in-memory settings."""
    from presentation.api.routes.admin_config import _apply_config_to_settings

    try:
        async with uow_factory.create(master=True) as uow:
            rows = await uow.config_parameters.get_all()
            for r in rows:
                _apply_config_to_settings(r.key, r.value, r.value_type)
            logger.info("Loaded %d config parameters from DB", len(rows))
    except Exception as e:
        logger.warning("Failed to load config from DB: %s", e)
