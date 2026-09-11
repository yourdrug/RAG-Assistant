"""Generic async exception handler decorator."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger("default")


def handle_exceptions(func: Callable) -> Callable:
    """Log exceptions without crashing the caller."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.exception(
                "Job %s failed: [%s] %s",
                func.__name__,
                type(e).__name__,
                e,
            )

    return wrapper
