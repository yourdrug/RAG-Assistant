"""Domain utilities — pure functions with no infrastructure dependencies."""

from __future__ import annotations

_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off"})


def parse_bool(raw: str) -> bool:
    """Parse a boolean string (case-insensitive).

    Accepted truthy: true, 1, yes, on
    Accepted falsy: false, 0, no, off
    Raises ValueError for unrecognized values.
    """
    lower = raw.lower().strip()
    if lower in _TRUE_VALUES:
        return True
    if lower in _FALSE_VALUES:
        return False
    raise ValueError(f"Cannot parse {raw!r} as bool")
