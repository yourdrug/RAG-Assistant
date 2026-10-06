"""Shared interpretation of benchmark source hints.

Semicolon-separated hints identify relevant sources. Hit Rate and MRR use
the first retrieved source matching any hint; blank hints provide no labels.
"""


def parse_source_hints(source_hint: str | None) -> tuple[str, ...]:
    """Normalize hints, ignoring whitespace and empty list entries."""
    if source_hint is None:
        return ()
    return tuple(part.strip().lower() for part in source_hint.split(";") if part.strip())


def matches_source_hints(filename: str, hints: tuple[str, ...]) -> bool:
    """Preserve case-insensitive substring matching for each individual hint."""
    return any(hint in filename.lower() for hint in hints)
