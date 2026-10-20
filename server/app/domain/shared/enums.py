"""ValidatedEnumMixin -- mixin that adds auto-generated validation error messages to StrEnum."""

from __future__ import annotations

from typing import Any, ClassVar, Self

from domain.shared.exceptions import ValidationError


class ValidatedEnumMixin:
    """Mixin for StrEnum subclasses that generates validation error messages.

    Must be listed *before* StrEnum in the MRO so that the ``validate``
    classmethod is available.  Example::

        class MyEnum(ValidatedEnumMixin, StrEnum, label="my_value"):
            A = "a"
            B = "b"

    The ``_label`` is set via ``__init_subclass__`` to avoid being
    treated as an enum member by the StrEnum metaclass.
    """

    _label: ClassVar[str] = "value"

    def __init_subclass__(cls, label: str = "value", **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._label = label

    @classmethod
    def validate(cls, value: str) -> Self:
        try:
            return cls(value)  # type: ignore[return-value, call-arg]
        except ValueError:
            allowed = ", ".join(v.value for v in cls)  # type: ignore[attr-defined]
            raise ValidationError(f"{cls._label} must be one of [{allowed}], got '{value}'") from None


class LifecycleStatusMixin:
    """Mixin for StrEnum status types that classifies states as terminal, active, or neither.

    Subclasses pass ``terminal=`` and ``active=`` as frozensets of string values.
    The properties compare ``self.value`` against these sets.
    """

    _terminal: ClassVar[frozenset[str]] = frozenset()
    _active: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(
        cls,
        terminal: frozenset[str] | None = None,
        active: frozenset[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init_subclass__(**kwargs)

        if terminal is not None:
            cls._terminal = terminal

        if active is not None:
            cls._active = active

    @property
    def is_terminal(self) -> bool:
        """True when no more work is expected for this state."""
        return self.value in self._terminal  # type: ignore[attr-defined]

    @property
    def is_active(self) -> bool:
        """True for PENDING, RUNNING, PROCESSING, INDEXING, IN_PROGRESS -- work is ongoing."""
        return self.value in self._active  # type: ignore[attr-defined]
