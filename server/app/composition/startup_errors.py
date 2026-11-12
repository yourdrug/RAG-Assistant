"""Readable startup failure for invalid configuration.

``config`` builds the settings singleton at import time and raises
``ConfigurationError`` instead of exiting, so tests and callers stay in control.
At a process entry point that exception is uncaught: Python would print a long
import traceback that buries the actual list of problems. This hook reports only
the problems (they never contain values) and lets the interpreter end the process
as for any uncaught exception; other exceptions keep the previous hook.

Importing this module installs the hook; ``composition/__init__.py`` imports it
before anything that loads ``config``.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from application.runtime.errors import ConfigurationError


if TYPE_CHECKING:
    from types import TracebackType


logger = logging.getLogger("config")


class ConfigurationErrorHook:
    def __init__(self, previous: object) -> None:
        self.previous = previous

    def __call__(
        self,
        exc_type: type[BaseException],
        exc: BaseException,
        tb: TracebackType | None,
    ) -> None:
        if isinstance(exc, ConfigurationError):
            # Logging may not be configured yet: ``logging.lastResort`` still writes CRITICAL to stderr.
            logger.critical("Invalid configuration, the process cannot start.\n%s", exc)
            return

        if callable(self.previous):
            self.previous(exc_type, exc, tb)


def install_configuration_error_hook() -> None:
    """Install once; keeps the hook that was active before as the fallback."""
    if not isinstance(sys.excepthook, ConfigurationErrorHook):
        sys.excepthook = ConfigurationErrorHook(sys.excepthook)


install_configuration_error_hook()


__all__ = ["ConfigurationErrorHook", "install_configuration_error_hook"]
