"""LifecycleStatusMixin -- mixin for status enums providing is_terminal / is_active properties.

Usage::

    from enum import StrEnum
    from domain.value_objects.lifecycle_status import LifecycleStatusMixin

    class MyStatus(LifecycleStatusMixin, StrEnum):
        _terminal = frozenset({"done", "failed"})
        _active = frozenset({"pending", "running"})
        PENDING = "pending"
        RUNNING = "running"
        DONE = "done"
        FAILED = "failed"
"""

from __future__ import annotations

from typing import ClassVar


class LifecycleStatusMixin:
    """Mixin for StrEnum status types that classifies states as terminal, active, or neither.

    Subclasses set ``_terminal`` and ``_active`` as frozensets of string values.
    The properties compare ``self.value`` against these sets.
    """

    _terminal: ClassVar[frozenset[str]] = frozenset()
    _active: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(
        cls,
        terminal: frozenset[str] | None = None,
        active: frozenset[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if terminal is not None:
            cls._terminal = terminal
        if active is not None:
            cls._active = active

    @property
    def is_terminal(self) -> bool:
        """True for DONE, FAILED, DEAD_LETTER, CANCELLED -- no further transitions expected."""
        return self.value in self._terminal

    @property
    def is_active(self) -> bool:
        """True for PENDING, RUNNING, PROCESSING, INDEXING, IN_PROGRESS -- work is ongoing."""
        return self.value in self._active
