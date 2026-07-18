"""
logging_configuration.py — конфигурация логирования для всего приложения.

Использование:
    import logging
    import logging_configuration
    logging.config.dictConfig(logging_config)
"""

import logging
from typing import Any

from logging_filters import ExceptionFilter, LevelMinFilter, LevelThresholdFilter

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
