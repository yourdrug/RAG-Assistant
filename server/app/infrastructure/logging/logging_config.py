"""Logging configuration -- dict-based config with custom filters.

Defines loggers (``default``, ``detailed``, ``uvicorn``), custom filters
(``ExceptionFilter``, ``LevelThresholdFilter``, ``LevelMinFilter``,
``RequestIDFilter``), and formatting.  Applied during application startup
via ``logging.config.dictConfig()``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from shared import request_id_ctx
from pythonjsonlogger.json import JsonFormatter


class ExceptionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.exc_info is None


class LevelThresholdFilter(logging.Filter):
    def __init__(self, max_level: int = logging.ERROR) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


class LevelMinFilter(logging.Filter):
    def __init__(self, min_level: int = logging.ERROR) -> None:
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level


class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("-")  # type: ignore[attr-defined]
        return True


LOG_FORMAT = os.getenv("LOG_FORMAT", "text")

_formatters: dict[str, Any] = {
    "default": {
        "()": "logging.Formatter",
        "format": "[%(asctime)s] %(levelname)s [%(request_id)s]: %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    },
    "detailed": {
        "()": "logging.Formatter",
        "format": "[%(asctime)s] %(levelname)s [%(request_id)s]: (%(filename)s %(lineno)d) - %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    },
    "uvicorn": {
        "()": "uvicorn.logging.DefaultFormatter",
        "fmt": "[%(asctime)s] %(levelname)s [%(request_id)s]: %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    },
    "access": {
        "()": "uvicorn.logging.AccessFormatter",
        "fmt": "[%(asctime)s] %(levelname)s [%(request_id)s]: %(client_addr)s %(request_line)s %(status_code)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    },
}

if LOG_FORMAT == "json":
    try:
        _json_formatter: dict[str, Any] = {
            "()": JsonFormatter,
            "format": "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s",
            "rename_fields": {"asctime": "timestamp", "levelname": "level", "name": "logger"},
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        }
        _formatters["json"] = _json_formatter
        _formatters["json_detailed"] = {
            **_json_formatter,
            "format": "%(asctime)s %(levelname)s %(name)s %(request_id)s %(filename)s %(lineno)d %(message)s",
        }
    except ImportError:
        LOG_FORMAT = "text"

_default_formatter = "json" if LOG_FORMAT == "json" else "default"
_detailed_formatter = "json_detailed" if LOG_FORMAT == "json" else "detailed"

logging_config: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "exclude_exceptions": {"()": ExceptionFilter},
        "below_error": {"()": LevelThresholdFilter, "max_level": logging.ERROR},
        "above_warning": {"()": LevelMinFilter, "min_level": logging.ERROR},
        "request_id": {"()": RequestIDFilter},
    },
    "formatters": _formatters,
    "handlers": {
        "default_stdout": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": _default_formatter,
            "stream": "ext://sys.stdout",
            "filters": ["below_error", "request_id"],
        },
        "default_stderr": {
            "class": "logging.StreamHandler",
            "level": "ERROR",
            "formatter": _default_formatter,
            "stream": "ext://sys.stderr",
            "filters": ["above_warning", "request_id"],
        },
        "detailed_stdout": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": _default_formatter,
            "stream": "ext://sys.stdout",
            "filters": ["below_error", "request_id"],
        },
        "detailed_stderr": {
            "class": "logging.StreamHandler",
            "level": "ERROR",
            "formatter": _detailed_formatter,
            "stream": "ext://sys.stderr",
            "filters": ["above_warning", "request_id"],
        },
        "uvicorn": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "uvicorn",
            "stream": "ext://sys.stderr",
            "filters": ["exclude_exceptions", "request_id"],
        },
        "access": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "access",
            "stream": "ext://sys.stdout",
            "filters": ["request_id"],
        },
        "null": {"class": "logging.NullHandler"},
    },
    "loggers": {
        "default": {"handlers": ["default_stdout", "default_stderr"], "level": "INFO", "propagate": False},
        "detailed": {"handlers": ["detailed_stdout", "detailed_stderr"], "level": "INFO", "propagate": False},
        "uvicorn": {"handlers": ["uvicorn"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["uvicorn"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
    "root": {"handlers": ["default_stdout", "default_stderr"], "level": "INFO"},
}
