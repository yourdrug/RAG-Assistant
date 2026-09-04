"""Top-level DI Container — orchestrates infrastructure + application.

Implementation lives in:
  - composition/infrastructure.py  → InfrastructureContainer (with sub-containers)
  - composition/application.py     → ApplicationContainer
  - composition/service_providers.py → factory methods for CLI/worker services
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from composition.application import ApplicationContainer
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
        Rolls back infrastructure resources if application init fails.
        """
        if self._initialized:
            raise RuntimeError(
                "Container.init() must be called exactly once. Second call detected — this is a bug."
            )
        self.infrastructure.init(database_manager)
        self._subscribe_config_events()
        try:
            self.application.init(self.infrastructure)
        except Exception:
            log.exception("ApplicationContainer.init() failed — rolling back infrastructure")
            self._unsubscribe_config_events()
            # Use synchronous dispose since application was never fully initialized
            # and infrastructure.dispose() has no awaitable work for DB cleanup.
            self.infrastructure._initialized = False
            raise
        self._initialized = True

        issues = self.infrastructure.validate()
        if issues:
            log.warning("Infrastructure validation issues: %s", issues)

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
        self._unsubscribe_config_events()
        self._initialized = False

    def _subscribe_config_events(self) -> None:
        """Subscribe config-change handlers to the event bus."""
        from domain.events.config_events import ConfigParameterChanged
        from infrastructure.events.in_process_event_bus import event_bus
        from infrastructure.ml.config_subscribers import (
            apply_to_settings,
            audit_log_config_change,
            invalidate_paddle_ocr_cache,
            invalidate_pii_detector_cache,
            invalidate_storage_cache,
        )

        ml = self.infrastructure.ml_clients  # raises ContainerNotInitializedError if not init'd

        bus = event_bus
        bus.subscribe(ConfigParameterChanged, apply_to_settings)
        bus.subscribe(ConfigParameterChanged, invalidate_paddle_ocr_cache)
        bus.subscribe(ConfigParameterChanged, invalidate_storage_cache)
        bus.subscribe(ConfigParameterChanged, invalidate_pii_detector_cache)
        bus.subscribe(ConfigParameterChanged, audit_log_config_change)

        def _invalidate_llm(event: ConfigParameterChanged) -> None:
            llm_keys = {
                "llm_provider",
                "llm_model",
                "llm_temperature",
                "llm_top_p",
                "llm_num_ctx_narrow",
                "llm_num_predict_narrow",
                "openrouter_model",
            }
            if event.key in llm_keys:
                ml.invalidate_llm()

        def _invalidate_bm25(event: ConfigParameterChanged) -> None:
            if event.key == "hybrid_enabled":
                ml.invalidate_bm25()

        bus.subscribe(ConfigParameterChanged, _invalidate_llm)
        bus.subscribe(ConfigParameterChanged, _invalidate_bm25)

    def _unsubscribe_config_events(self) -> None:
        """Remove all config-change handlers from the event bus."""
        from infrastructure.events.in_process_event_bus import event_bus

        event_bus.unsubscribe_all()
