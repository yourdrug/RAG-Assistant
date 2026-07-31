"""Shared utilities for route modules."""

from __future__ import annotations

import logging

logger = logging.getLogger("default")


def safe_background_call(func, *args, **kwargs):
    """Run a sync function in a background task, catching and logging exceptions."""
    try:
        func(*args, **kwargs)
    except Exception:
        logger.exception("Background task failed")
