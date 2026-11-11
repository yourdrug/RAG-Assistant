"""Bound structural paths while retaining their most specific level."""


def shorten_breadcrumb(path: str, limit: int) -> str:
    if len(path) <= limit:
        return path

    levels = path.split(" > ")
    tail = levels.pop()

    if len(tail) > limit:
        return tail[: max(0, limit - 1)].rstrip() + "…"

    while levels and len(" > ".join([levels[-1], tail])) + (4 if len(levels) > 1 else 0) <= limit:
        tail = levels.pop() + " > " + tail

    return ("… > " + tail) if levels and len(tail) + 4 <= limit else tail
