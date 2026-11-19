"""Ownership of secret, static, metadata and dynamic setting keys.

Secret/static/metadata sets mirror the ``Secret``/``Static``/``Metadata`` markers on
settings section fields (``config.ownership``); Application cannot import ``config``,
so ``tests/test_configuration_sources.py`` enforces that both stay identical.
"""

from __future__ import annotations

from application.runtime.dynamic_params import DYNAMIC_PARAMS


# Ownership is exclusive; dynamic keys derive from the canonical declaration registry.
SECRET_SETTING_KEYS: frozenset[str] = frozenset(
    {
        "db_password",
        "qdrant_api_key",
        "llm_api_key",
        "embedding_api_key",
        "rerank_api_key",
        "redis_password",
        "s3_access_key",
        "s3_secret_key",
        "jwt_secret_key",
        "admin_password",
        "metrics_scrape_token",
    }
)

STATIC_SETTING_KEYS: frozenset[str] = frozenset(
    {
        "ml_internal_allowed_origins",
        "llm_allowed_origins",
        "ml_allowed_origins",
        "ocr_vl_url",
        "ocr_vl_timeout",
        "db_host",
        "db_port",
        "db_user",
        "db_name",
        "db_use_pgbouncer",
        "db_direct_host",
        "db_direct_port",
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
        "benchmark_llm_replay_retention_days",
        "benchmark_llm_replay_epoch",
        "embedding_model",
        "embedding_revision",
        "embed_dim",
        "embed_tokenizer_path",
        "chunk_contextualization_enabled",
        "chunk_contextualization_concurrency",
        "chunk_contextualization_timeout",
        "curator_scope_max_ids",
        "otel_enabled",
        "otel_exporter_endpoint",
        "redis_host",
        "redis_port",
        "redis_user",
        "redis_db",
        "worker_max_concurrent",
        "maintenance_max_concurrent",
        "document_max_concurrent",
        "job_max_per_principal",
        "job_backlog_limit",
        "job_principal_backlog_limit",
        "worker_completion_wait_seconds",
        "ingest_max_concurrent",
        "benchmark_job_max_concurrent",
        "sweep_max_concurrent",
        "outbox_max_concurrent",
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


GLOBAL_DYNAMIC_SETTING_KEYS: frozenset[str] = frozenset(
    param.key for param in DYNAMIC_PARAMS if param.domain_key is None
)
