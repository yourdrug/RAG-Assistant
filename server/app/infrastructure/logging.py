"""
infrastructure/logging.py — Custom logging filters + logging config dict.
Merged from logging_filters.py and logging_configuration.py.
"""

import logging
from typing import Any

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class ExceptionFilter(logging.Filter):
    """Исключает записи с exception info (traceback от uvicorn)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.exc_info is None


class LevelThresholdFilter(logging.Filter):
    """Пропускает записи УРОВНЯ <= max_level."""

    def __init__(self, max_level: int = logging.ERROR) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


class LevelMinFilter(logging.Filter):
    """Пропускает записи УРОВНЯ >= min_level."""

    def __init__(self, min_level: int = logging.ERROR) -> None:
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level


# ---------------------------------------------------------------------------
# Config dict
# ---------------------------------------------------------------------------

logging_config: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "exclude_exceptions": {
            "()": ExceptionFilter,
        },
        "below_error": {
            "()": LevelThresholdFilter,
            "max_level": logging.ERROR,
        },
        "above_warning": {
            "()": LevelMinFilter,
            "min_level": logging.ERROR,
        },
    },
    "formatters": {
        "default": {
            "()": "logging.Formatter",
            "format": "[%(asctime)s] %(levelname)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "()": "logging.Formatter",
            "format": "[%(asctime)s] %(levelname)s: (%(filename)s %(lineno)d) - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "uvicorn": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "[%(asctime)s] %(levelname)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": "[%(asctime)s] %(levelname)s: %(client_addr)s %(request_line)s %(status_code)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default_stdout": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout",
            "filters": ["below_error"],
        },
        "default_stderr": {
            "class": "logging.StreamHandler",
            "level": "ERROR",
            "formatter": "default",
            "stream": "ext://sys.stderr",
            "filters": ["above_warning"],
        },
        "detailed_stdout": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout",
            "filters": ["below_error"],
        },
        "detailed_stderr": {
            "class": "logging.StreamHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "stream": "ext://sys.stderr",
            "filters": ["above_warning"],
        },
        "uvicorn": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "uvicorn",
            "stream": "ext://sys.stderr",
            "filters": ["exclude_exceptions"],
        },
        "access": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "access",
            "stream": "ext://sys.stdout",
        },
        "null": {
            "class": "logging.NullHandler",
        },
    },
    "loggers": {
        "default": {
            "handlers": ["default_stdout", "default_stderr"],
            "level": "INFO",
            "propagate": False,
        },
        "detailed": {
            "handlers": ["detailed_stdout", "detailed_stderr"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn": {"handlers": ["uvicorn"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["uvicorn"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
    "root": {"handlers": ["default_stdout", "default_stderr"], "level": "INFO"},
}
