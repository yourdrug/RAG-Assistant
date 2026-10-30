"""Ownership of secret, static, metadata and dynamic setting keys."""

from __future__ import annotations

from application.runtime.dynamic_params import DYNAMIC_PARAMS


# Ownership is exclusive; dynamic keys derive from the canonical declaration registry.
SECRET_SETTING_KEYS: frozenset[str] = frozenset(
    {
        "db_password",
        "qdrant_api_key",
        "openrouter_api_key",
        "deepinfra_api_key",
        "redis_password",
        "s3_access_key",
        "s3_secret_key",
        "jwt_secret_key",
        "admin_password",
    }
)

STATIC_SETTING_KEYS: frozenset[str] = frozenset(
    {
        "evidence_profile",
        "ml_internal_allowed_origins",
        "ocr_vl_url",
        "ocr_vl_timeout",
        "db_host",
        "db_port",
        "db_user",
        "db_name",
        "db_slave_hosts",
        "db_slave_ports",
        "db_pool_size",
        "db_max_overflow",
        "db_pool_recycle_sec",
        "db_statement_timeout_ms",
        "db_idle_in_transaction_timeout_ms",
        "qdrant_timeout",
        "qdrant_search_max_concurrent",
        "qdrant_write_max_concurrent",
        "rag_retrieval_score_threshold",
        "llm_max_concurrent",
        "llm_generation_max_concurrent",
        "llm_auxiliary_max_concurrent",
        "llm_request_timeout",
        "llm_auxiliary_timeout",
        "embedding_max_concurrent",
        "ingestion_max_concurrent",
        "reranker_max_concurrent",
        "semaphore_acquire_timeout",
        "benchmark_max_concurrent",
        "benchmark_judge_max_concurrent",
        "benchmark_judge_grouped_enabled",
        "benchmark_judge_initial_tokens",
        "benchmark_judge_retry_tokens",
        "benchmark_judge_model",
        "tei_embed_model",
        "embedding_revision",
        "embed_dim",
        "curator_scope_max_ids",
        "redis_host",
        "redis_port",
        "redis_user",
        "redis_db",
        "worker_max_concurrent",
        "redis_socket_timeout",
        "redis_max_connections",
        "s3_endpoint",
        "s3_bucket",
        "s3_region",
        "s3_max_concurrent",
        "supported_extensions",
        "data_dir",
        "uploads_prefix",
        "allowed_origins",
        "forwarded_allow_ips",
        "jwt_algorithm",
        "jwt_expire_minutes",
        "admin_email",
        "timezone",
        "stage",
        "rate_limit_redis_prefix",
        "rate_limit_max_buckets",
        "llm_breaker_fail_max",
        "llm_breaker_timeout_duration",
    }
)

METADATA_SETTING_KEYS: frozenset[str] = frozenset(
    {
        "version",
        "service_start_datetime",
    }
)

TARGET_DYNAMIC_SETTING_KEYS: frozenset[str] = frozenset(param.key for param in DYNAMIC_PARAMS)
