"""
logging_filters.py — кастомные фильтры для логирования.
"""

import logging


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
