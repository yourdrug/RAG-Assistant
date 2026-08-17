"""Shared dependency providers for Arq worker tasks.

The Arq worker runs in a separate process and cannot use FastAPI's DI.
This module provides module-level singletons that the worker's ``_on_startup``
callback registers at process start.

IMPORTANT: This module exists ONLY for Arq worker tasks. The main API process
uses ``presentation/api/dependencies.py`` as its Composition Root. The scheduler
uses instance-injected dependencies — do NOT add scheduler usage here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from application.ports.unit_of_work_factory import UnitOfWorkFactory

# ---- UoW factory (worker-only) ----
_uow_factory: UnitOfWorkFactory | None = None


def register_uow_factory(factory: UnitOfWorkFactory) -> None:
    global _uow_factory
    _uow_factory = factory


def get_uow_factory() -> UnitOfWorkFactory:
    if _uow_factory is None:
        raise RuntimeError("UnitOfWorkFactory not registered. Call register_uow_factory() at startup.")
    return _uow_factory


# ---- Config listener (worker-only) ----
_config_listener = None


def register_config_listener(listener) -> None:
    global _config_listener
    _config_listener = listener


def get_config_listener():
    if _config_listener is None:
        raise RuntimeError("Config listener not registered. Call register_config_listener() at startup.")
    return _config_listener
