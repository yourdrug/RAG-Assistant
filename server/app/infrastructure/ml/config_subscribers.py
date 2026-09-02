"""ConfigParameterChanged subscribers -- hot-reload reactions for dynamic settings.

Each concern is a separate function; adding a new reaction means adding a new
subscribe() call without touching existing ones.

Dynamic (hot-reloadable) parameters are those listed in _DYNAMIC_FIELDS.
They can be changed via the API without a restart and are applied to the
in-memory settings immediately after a DB commit + NOTIFY.

Static parameters -- everything else in .env / Settings -- cannot be changed
without restarting the process (read once at startup).  When adding a new
parameter, decide: should it be hot-reloadable?  If yes, add it to
_DYNAMIC_FIELDS and the config_parameters DB table.  If no, leave it in
.env and require a restart.
"""

from __future__ import annotations

import json
import logging

from config import settings
from domain.events.config_events import ConfigParameterChanged
from domain.utils import parse_bool

from infrastructure.ml.ingestion import _get_paddle_ocr
from infrastructure.storage import get_storage

log = logging.getLogger("default")

SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "s3_access_key",
        "s3_secret_key",
        "openrouter_api_key",
        "deepinfra_api_key",
        "jwt_secret_key",
        "db_password",
        "redis_password",
    }
)


def _mask_value(value: str | None) -> str:
    """Return masked representation for sensitive values."""
    if value is None:
        return "None"
    if len(value) <= 4:
        return "****"
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


_DYNAMIC_FIELDS: dict[str, tuple[str, type]] = {
    # --- RAG ---
    "retriever_fetch_k": ("retriever_fetch_k", int),
    "retriever_top_k": ("retriever_top_k", int),
    "retriever_fetch_k_broad": ("retriever_fetch_k_broad", int),
    "retriever_top_k_broad": ("retriever_top_k_broad", int),
    "history_window": ("history_window", int),
    "chunk_size": ("chunk_size", int),
    "chunk_overlap": ("chunk_overlap", int),
    "source_min_score": ("source_min_score", float),
    # --- Hybrid search ---
    "hybrid_enabled": ("hybrid_enabled", bool),
    "bm25_fetch_k": ("bm25_fetch_k", int),
    "rrf_k": ("rrf_k", int),
    "dense_weight": ("dense_weight", float),
    "sparse_weight": ("sparse_weight", float),
    # --- Reranker / query-time filters ---
    "rerank_min_score": ("rerank_min_score", float),
    "rerank_score_gap_ratio": ("rerank_score_gap_ratio", float),
    "citation_filter_enabled": ("citation_filter_enabled", bool),
    "exact_ref_sparse_boost": ("exact_ref_sparse_boost", float),
    # --- Ingestion ---
    "embed_batch_size": ("embed_batch_size", int),
    # --- Relevance gate ---
    "relevance_gate_enabled": ("relevance_gate_enabled", bool),
    # --- Condense (rewrite follow-up questions) ---
    "condense_enabled": ("condense_enabled", bool),
    # --- Decomposition ---
    "decomposition_enabled": ("decomposition_enabled", bool),
    # --- Rolling summary ---
    "rolling_summary_enabled": ("rolling_summary_enabled", bool),
    # --- Cache ---
    "cache_enabled": ("cache_enabled", bool),
    # --- PII guardrail ---
    "pii_redaction_enabled": ("pii_redaction_enabled", bool),
    # --- LLM ---
    "llm_provider": ("llm_provider", str),
    "llm_model": ("llm_model", str),
    "llm_temperature": ("llm_temperature", float),
    "llm_top_p": ("llm_top_p", float),
    "llm_num_ctx_narrow": ("llm_num_ctx_narrow", int),
    "llm_num_ctx_broad": ("llm_num_ctx_broad", int),
    "llm_num_predict_narrow": ("llm_num_predict_narrow", int),
    "llm_num_predict_broad": ("llm_num_predict_broad", int),
    # --- OpenRouter ---
    "openrouter_model": ("openrouter_model", str),
    # --- ML Provider ---
    "ml_provider": ("ml_provider", str),
    "deepinfra_embed_model": ("deepinfra_embed_model", str),
    "deepinfra_rerank_model": ("deepinfra_rerank_model", str),
    # --- OCR ---
    "ocr_enabled": ("ocr_enabled", bool),
    "ocr_engine": ("ocr_engine", str),
    "ocr_dpi": ("ocr_dpi", int),
    "ocr_min_chars": ("ocr_min_chars", int),
    "ocr_lang_surya": ("ocr_lang_surya", list),
    "ocr_lang_paddle": ("ocr_lang_paddle", str),
    # --- Storage (s3_endpoint..region are dynamic via cache invalidation) ---
    "s3_endpoint": ("s3_endpoint", str),
    "s3_bucket": ("s3_bucket", str),
    "s3_region": ("s3_region", str),
}


def _coerce_and_set(attr: str, expected_type: type | None, new_value: str) -> None:
    """Coerce *new_value* to *expected_type* and set it on the global settings."""
    if expected_type is bool:
        setattr(settings, attr, parse_bool(new_value))
    elif expected_type is int:
        setattr(settings, attr, int(new_value))
    elif expected_type is float:
        setattr(settings, attr, float(new_value))
    elif expected_type is list:
        setattr(settings, attr, json.loads(new_value))
    elif expected_type is str:
        raw = new_value
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, str):
                raw = parsed
        except (json.JSONDecodeError, TypeError):
            pass
        setattr(settings, attr, raw)
    else:
        setattr(settings, attr, new_value)


def apply_to_settings(event: ConfigParameterChanged) -> None:
    """Применить новое значение к in-memory settings."""
    if event.key in SENSITIVE_KEYS:
        return
    attr, expected_type = _DYNAMIC_FIELDS.get(event.key, (event.key, None))
    if not hasattr(settings, attr):
        return
    try:
        _coerce_and_set(attr, expected_type, event.new_value)
        log.info("Config applied: %s = %s (was %s)", event.key, event.new_value, event.old_value)
    except (ValueError, TypeError) as e:
        log.warning("Failed to apply config %s=%r: %s", event.key, event.new_value, e)


# ---------------------------------------------------------------------------
# Cache invalidation subscribers
# ---------------------------------------------------------------------------


def invalidate_paddle_ocr_cache(event: ConfigParameterChanged) -> None:
    """Сбросить кэш PaddleOCR при смене языка (модель перезагрузится лениво)."""
    if event.key != "ocr_lang_paddle":
        return

    _get_paddle_ocr.cache_clear()
    log.info("PaddleOCR cache invalidated (ocr_lang_paddle -> %s)", event.new_value)


def invalidate_storage_cache(event: ConfigParameterChanged) -> None:
    """Сбросить кэш хранилища при изменении backend или S3-параметров."""
    storage_keys = {"file_backend", "s3_endpoint", "s3_bucket", "s3_access_key", "s3_secret_key", "s3_region"}
    if event.key not in storage_keys:
        return

    get_storage.cache_clear()
    if event.key in SENSITIVE_KEYS:
        log.info("Storage cache invalidated (%s -> %s)", event.key, _mask_value(event.new_value))
    else:
        log.info("Storage cache invalidated (%s -> %s)", event.key, event.new_value)


def audit_log_config_change(event: ConfigParameterChanged) -> None:
    """Независимый аудит-лог — не зависит от settings/кэшей."""
    if event.key in SENSITIVE_KEYS:
        log.info(
            "AUDIT config_change key=%s old=%s new=%s by_user=%s at=%s",
            event.key,
            _mask_value(event.old_value),
            _mask_value(event.new_value),
            event.changed_by,
            event.occurred_at,
        )
    else:
        log.info(
            "AUDIT config_change key=%s old=%r new=%r by_user=%s at=%s",
            event.key,
            event.old_value,
            event.new_value,
            event.changed_by,
            event.occurred_at,
        )


def invalidate_pii_detector_cache(event: ConfigParameterChanged) -> None:
    """Сбросить кэш PII-детектора при изменении pii_redaction_enabled."""
    if event.key != "pii_redaction_enabled":
        return

    from infrastructure.ml.guardrails import invalidate_pii_detector

    invalidate_pii_detector()
    log.info("PII detector cache invalidated (pii_redaction_enabled -> %s)", event.new_value)
