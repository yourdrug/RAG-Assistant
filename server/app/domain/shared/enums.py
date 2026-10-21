"""String enums with validation and lifecycle classification."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Self

from domain.shared.exceptions import ValidationError


class ValidatedStrEnum(StrEnum):
    """String enum base that generates validation error messages.

    Example::

        class MyEnum(ValidatedStrEnum, label="my_value"):
            A = "a"
            B = "b"

    The ``_label`` is set via ``__init_subclass__`` to avoid being
    treated as an enum member by the StrEnum metaclass.
    """

    _label: ClassVar[str]

    def __init_subclass__(cls, label: str = "value", **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._label = label

    @classmethod
    def validate(cls, value: str) -> Self:
        try:
            return cls(value)
        except ValueError:
            allowed = ", ".join(v.value for v in cls)
            msg = f"{cls._label} must be one of [{allowed}], got '{value}'"
            raise ValidationError(msg) from None


class LifecycleStatus(StrEnum):
    """String enum base that classifies states as terminal, active, or neither.

    Subclasses pass ``terminal=`` and ``active=`` as frozensets of string values.
    The properties compare ``self.value`` against these sets.
    """

    _value_: str
    _terminal: ClassVar[frozenset[str]]
    _active: ClassVar[frozenset[str]]

    def __init_subclass__(
        cls,
        terminal: frozenset[str] | None = None,
        active: frozenset[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init_subclass__(**kwargs)

        if terminal is not None or not hasattr(cls, "_terminal"):
            cls._terminal = terminal if terminal is not None else frozenset()

        if active is not None or not hasattr(cls, "_active"):
            cls._active = active if active is not None else frozenset()

    @property
    def is_terminal(self) -> bool:
        """True when no more work is expected for this state."""
        return self.value in self._terminal

    @property
    def is_active(self) -> bool:
        """True for PENDING, RUNNING, PROCESSING, INDEXING, IN_PROGRESS -- work is ongoing."""
        return self.value in self._active
