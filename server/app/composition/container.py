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
from config import settings
from domain.value_objects.app_stage import AppStage

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
        try:
            self.application.init(self.infrastructure)
        except Exception:
            log.exception("ApplicationContainer.init() failed — rolling back infrastructure")
            self.unsubscribe_config_events()
            # Use synchronous dispose since application was never fully initialized
            # and infrastructure.dispose() has no awaitable work for DB cleanup.
            self.infrastructure._initialized = False
            raise
        self._initialized = True

        issues = self.infrastructure.validate()
        if issues:
            msg = "Infrastructure validation issues: %s" % issues
            if settings.stage == AppStage.PROD:
                raise RuntimeError(msg)
            log.warning(msg)

        app_issues = self.application.validate()
        if app_issues:
            msg = "Application validation issues: %s" % app_issues
            if settings.stage == AppStage.PROD:
                raise RuntimeError(msg)
            log.warning(msg)

        log.info(
            "Container initialized: %d infrastructure + %d application objects",
            len(dataclasses.fields(self.infrastructure)),
            len(dataclasses.fields(self.application)),
        )

    async def dispose(self) -> None:
        """Tear down all resources in reverse order.

        Safe to call even if init() was never called.
        """
        try:
            await self.application.dispose()
        finally:
            try:
                await self.infrastructure.dispose()
            finally:
                self.unsubscribe_config_events()
                self._initialized = False

    def subscribe_config_events(self) -> None:
        self.infrastructure.subscribe_config_events()

    def unsubscribe_config_events(self) -> None:
        self.infrastructure.unsubscribe_config_events()
