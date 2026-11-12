"""Configuration failure shared by the config package and process entry points."""

from __future__ import annotations


class ConfigurationError(Exception):
    """Invalid configuration; raised instead of exiting so callers and tests decide.

    Not a ``ValueError``: pydantic propagates it unwrapped, so the error never
    carries ``input_value`` with the loaded (secret) settings. It lives outside
    ``config`` so a startup hook can recognise it although importing ``config``
    is exactly what failed.
    """


__all__ = ["ConfigurationError"]
