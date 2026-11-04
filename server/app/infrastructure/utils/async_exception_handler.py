"""Generic async exception handler decorator."""

from __future__ import annotations

from functools import wraps
import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

logger = logging.getLogger("default")


def handle_exceptions[**P, R](
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Coroutine[object, object, R | None]]:
    """Log exceptions without crashing the caller."""

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        try:
            return await func(*args, **kwargs)
        except Exception:
            logger.exception("Job %s failed", func.__name__)
            return None

    return wrapper
