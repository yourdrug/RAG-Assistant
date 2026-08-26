"""Top-level DI Container — orchestrates infrastructure + application.

Implementation lives in:
  - composition/infrastructure.py  → InfrastructureContainer
  - composition/application.py     → ApplicationContainer, _ChunkSearchAdapter
  - composition/events.py          → _subscribe_config_events, _unsubscribe_config_events
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from composition.application import ApplicationContainer
from composition.events import _subscribe_config_events, _unsubscribe_config_events
from composition.infrastructure import InfrastructureContainer

if TYPE_CHECKING:
    from infrastructure.database.database import DatabaseManager

log = logging.getLogger("default")


@dataclass
class Container:
    """Top-level DI container.

    Usage::

        container = Container()
        container.init(database_manager)
        app.state.container = container   # for request-scoped access
        # ... serve requests ...
        await container.dispose()
    """

    infrastructure: InfrastructureContainer = field(default_factory=InfrastructureContainer)
    application: ApplicationContainer = field(default_factory=ApplicationContainer)
    _initialized: bool = field(default=False, repr=False)

    def init(self, database_manager: DatabaseManager) -> None:
        """Build the entire dependency graph.

        Must be called exactly once per process.
        Raises RuntimeError if called more than once.
        """
        if self._initialized:
            raise RuntimeError(
                "Container.init() must be called exactly once. Second call detected — this is a bug."
            )
        self.infrastructure.init(database_manager)
        self.application.init(self.infrastructure)
        _subscribe_config_events(self.infrastructure)
        self._initialized = True
        log.info(
            "Container initialized: %d infrastructure + %d application objects",
            len(dataclasses.fields(self.infrastructure)),
            len(dataclasses.fields(self.application)),
        )

    async def dispose(self) -> None:
        """Tear down all resources in reverse order.

        Safe to call even if init() was never called.
        """
        if not self._initialized:
            return
        await self.application.dispose()
        await self.infrastructure.dispose()
        _unsubscribe_config_events()
        self._clear_global_caches()
        self._initialized = False

    @staticmethod
    def _clear_global_caches() -> None:
        """Clear process-global lru_cache singletons so a fresh Container gets clean state.

        Avoids stale caches leaking across container re-creation.
        """
        from infrastructure.ml.ingestion import _get_paddle_ocr, _get_surya_predictors
        from infrastructure.storage.file_storage import get_storage

        get_storage.cache_clear()
        _get_paddle_ocr.cache_clear()
        _get_surya_predictors.cache_clear()
