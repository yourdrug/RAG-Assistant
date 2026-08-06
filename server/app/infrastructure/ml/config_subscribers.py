"""Подписчики на ConfigParameterChanged.

Каждая забота — отдельная функция; добавить новую реакцию = добавить
новый subscribe(), не трогая существующие.

Динамические (hot-reloadable) параметры — те, что перечислены в _DYNAMIC_FIELDS.
Их можно менять через API без перезапуска; они применяются к in-memory settings
сразу после commit в БД + NOTIFY.

Статические параметры — всё остальное в .env / Settings.
Их нельзя изменить без перезапуска процесса (читаются один раз при старте).
При добавлении нового параметра решите: должен ли он быть hot-reloadable?
Если да — добавьте его в _DYNAMIC_FIELDS и в БД-таблицу config_parameters.
Если нет — он остаётся в .env и требует рестарта.
"""

from __future__ import annotations

import logging

from config import settings
from domain.events.config_events import ConfigParameterChanged
from domain.utils import parse_bool

log = logging.getLogger("default")

_DYNAMIC_FIELDS: dict[str, tuple[str, type]] = {
    "retriever_fetch_k": ("retriever_fetch_k", int),
    "retriever_top_k": ("retriever_top_k", int),
    "retriever_fetch_k_broad": ("retriever_fetch_k_broad", int),
    "retriever_top_k_broad": ("retriever_top_k_broad", int),
    "history_window": ("history_window", int),
    "chunk_size": ("chunk_size", int),
    "chunk_overlap": ("chunk_overlap", int),
    "hybrid_enabled": ("hybrid_enabled", bool),
    "bm25_fetch_k": ("bm25_fetch_k", int),
    "rrf_k": ("rrf_k", int),
    "dense_weight": ("dense_weight", float),
    "sparse_weight": ("sparse_weight", float),
    "embed_batch_size": ("embed_batch_size", int),
    "source_min_score": ("source_min_score", float),
}


def apply_to_settings(event: ConfigParameterChanged) -> None:
    """Применить новое значение к in-memory settings."""
    attr, expected_type = _DYNAMIC_FIELDS.get(event.key, (event.key, None))
    if not hasattr(settings, attr):
        return
    try:
        if expected_type is bool:
            setattr(settings, attr, parse_bool(event.new_value))
        elif expected_type is int:
            setattr(settings, attr, int(event.new_value))
        elif expected_type is float:
            setattr(settings, attr, float(event.new_value))
        else:
            setattr(settings, attr, event.new_value)
        log.info("Config applied: %s = %s (was %s)", event.key, event.new_value, event.old_value)
    except (ValueError, TypeError) as e:
        log.warning("Failed to apply config %s=%r: %s", event.key, event.new_value, e)


def invalidate_bm25_cache_on_hybrid_toggle(event: ConfigParameterChanged) -> None:
    """Если переключили hybrid_enabled — сбросить lru_cache BM25-индекса."""
    if event.key != "hybrid_enabled":
        return
    from infrastructure.clients import get_bm25_index

    get_bm25_index.cache_clear()
    log.info("BM25 index cache invalidated (hybrid_enabled -> %s)", event.new_value)


def audit_log_config_change(event: ConfigParameterChanged) -> None:
    """Независимый аудит-лог — не зависит от settings/кэшей."""
    log.info(
        "AUDIT config_change key=%s old=%r new=%r by_user=%s at=%s",
        event.key,
        event.old_value,
        event.new_value,
        event.changed_by,
        event.occurred_at,
    )
