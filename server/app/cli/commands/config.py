"""CLI-команда: ручной resync динамической конфигурации из БД.

Отличие от listener resync:
- Одноразовая команда (не слушает NOTIFY)
- Не отправляет pg_notify другим процессам (они не узнают об изменении)
- Использует ту же логику сравнения с текущими settings, что и listener
"""

from __future__ import annotations

import asyncio
import logging
import sys

import typer

logger = logging.getLogger("cli")

config_app = typer.Typer(help="Управление динамической конфигурацией")


@config_app.command("resync")
def config_resync() -> None:
    """Принудительно применить все config_parameters из БД к in-memory settings.

    Полезно после ручного изменения таблицы config_parameters в обход API,
    либо для диагностики, если /health показывает config_listener: error.
    """
    from domain.events.config_events import ConfigParameterChanged
    from infrastructure.database.database import database
    from infrastructure.events.in_process_event_bus import event_bus
    from infrastructure.ml.config_subscribers import apply_to_settings
    from infrastructure.uow_factory import UnitOfWorkFactory

    event_bus.subscribe(ConfigParameterChanged, apply_to_settings)

    async def _run() -> None:
        from config import settings

        await database.connect()
        uow_factory = UnitOfWorkFactory(database=database)

        try:
            async with uow_factory.create() as uow:
                rows = await uow.config_parameters.get_all()

            applied = 0
            for r in rows:
                current = getattr(settings, r.key, None)
                current_str = (
                    str(current).lower()
                    if isinstance(current, bool)
                    else (str(current) if current is not None else None)
                )
                if current_str == r.value:
                    continue
                event_bus.publish(
                    ConfigParameterChanged(
                        key=r.key,
                        old_value=current_str,
                        new_value=r.value,
                        value_type=r.value_type,
                    )
                )
                applied += 1
            logger.info("Resynced %d/%d parameters (changed only)", applied, len(rows))
        finally:
            await database.disconnect()

    try:
        asyncio.run(_run())
    except Exception as exc:
        logger.error("Config resync failed", exc_info=exc)
        sys.exit(1)
