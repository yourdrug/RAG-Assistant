"""Interpret the legacy source labels independently of retrieval technology."""


def parse_source_hints(source_hint: str | None) -> tuple[str, ...]:
    if source_hint is None:
        return ()
    return tuple(part.strip().lower() for part in source_hint.split(";") if part.strip())


def matches_source_hints(filename: str, hints: tuple[str, ...]) -> bool:
    return any(hint in filename.lower() for hint in hints)
