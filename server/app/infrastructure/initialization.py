"""Application bootstrap -- creates the default admin user and loads dynamic config from DB.

Called once during FastAPI lifespan startup.  If the admin user already
exists, the step is a no-op.  Dynamic config parameters are loaded from the
``config_parameters`` table and applied to in-memory settings.  If the table
is empty, default values are seeded from the current env-based settings.
"""

from __future__ import annotations

import json
import logging

from config import settings
from domain.entities.config_parameter import ConfigParameter
from domain.events.config_events import ConfigParameterChanged
from domain.value_objects.roles import UserKind, UserRole
from infrastructure.auth.password_hasher import BCryptPasswordHasher
from infrastructure.events.in_process_event_bus import event_bus

logger = logging.getLogger("default")


def _param(key, value, vtype, cat, desc, min_v=None, max_v=None, allowed=None):
    """Build a ConfigParameter entity for seeding from env defaults."""
    return ConfigParameter(
        key=key,
        value=value,
        value_type=vtype,
        category=cat,
        description=desc,
        min_value=min_v,
        max_value=max_v,
        allowed_values=allowed,
    )


def _build_defaults() -> list[ConfigParameter]:
    """Build all default config parameters from current env settings."""
    s = settings
    return [
        # --- RAG ---
        _param(
            "retriever_fetch_k",
            str(s.retriever_fetch_k),
            "int",
            "rag",
            "Retriever fetch count (narrow)",
            1,
            200,
        ),
        _param("retriever_top_k", str(s.retriever_top_k), "int", "rag", "Retriever top-k (narrow)", 1, 50),
        _param(
            "retriever_fetch_k_broad",
            str(s.retriever_fetch_k_broad),
            "int",
            "rag",
            "Retriever fetch count (broad)",
            1,
            200,
        ),
        _param(
            "retriever_top_k_broad",
            str(s.retriever_top_k_broad),
            "int",
            "rag",
            "Retriever top-k (broad)",
            1,
            50,
        ),
        _param("history_window", str(s.history_window), "int", "rag", "Chat history window", 0, 50),
        _param("chunk_size", str(s.chunk_size), "int", "rag", "Document chunk size (chars)", 100, 5000),
        _param("chunk_overlap", str(s.chunk_overlap), "int", "rag", "Chunk overlap (chars)", 0, 1000),
        _param(
            "source_min_score",
            str(s.source_min_score),
            "float",
            "rag",
            "Minimum source relevance score",
            0.0,
            1.0,
        ),
        # --- Hybrid search ---
        _param(
            "hybrid_enabled",
            json.dumps(s.hybrid_enabled),
            "bool",
            "hybrid",
            "Enable hybrid search (dense+sparse)",
        ),
        _param("bm25_fetch_k", str(s.bm25_fetch_k), "int", "hybrid", "BM25 fetch count", 1, 200),
        _param("rrf_k", str(s.rrf_k), "int", "hybrid", "RRF fusion parameter", 1, 200),
        _param("dense_weight", str(s.dense_weight), "float", "hybrid", "Dense vector weight", 0.0, 10.0),
        _param("sparse_weight", str(s.sparse_weight), "float", "hybrid", "Sparse vector weight", 0.0, 10.0),
        # --- Reranker ---
        _param(
            "rerank_min_score",
            str(s.rerank_min_score) if s.rerank_min_score is not None else "0.15",
            "float",
            "reranker",
            "Minimum reranker score threshold",
            0.0,
            1.0,
        ),
        _param(
            "rerank_score_gap_ratio",
            str(s.rerank_score_gap_ratio) if s.rerank_score_gap_ratio is not None else "0.1",
            "float",
            "reranker",
            "Reranker score gap ratio",
            0.0,
            1.0,
        ),
        _param(
            "citation_filter_enabled",
            json.dumps(s.citation_filter_enabled),
            "bool",
            "reranker",
            "Enable citation filter",
        ),
        _param(
            "exact_ref_sparse_boost",
            str(s.exact_ref_sparse_boost),
            "float",
            "reranker",
            "Exact reference sparse boost",
            0.0,
            10.0,
        ),
        # --- Ingestion ---
        _param(
            "embed_batch_size", str(s.embed_batch_size), "int", "ingestion", "Embedding batch size", 1, 128
        ),
        # --- Feature toggles ---
        _param(
            "relevance_gate_enabled",
            json.dumps(s.relevance_gate_enabled),
            "bool",
            "toggles",
            "Enable relevance gate",
        ),
        _param(
            "condense_enabled",
            json.dumps(s.condense_enabled),
            "bool",
            "toggles",
            "Enable follow-up question condensing",
        ),
        _param(
            "decomposition_enabled",
            json.dumps(s.decomposition_enabled),
            "bool",
            "toggles",
            "Enable query decomposition",
        ),
        _param(
            "rolling_summary_enabled",
            json.dumps(s.rolling_summary_enabled),
            "bool",
            "toggles",
            "Enable rolling summary",
        ),
        _param("cache_enabled", json.dumps(s.cache_enabled), "bool", "toggles", "Enable response cache"),
        _param(
            "pii_redaction_enabled",
            json.dumps(s.pii_redaction_enabled),
            "bool",
            "toggles",
            "Enable PII redaction",
        ),
        # --- LLM ---
        _param(
            "llm_provider",
            s.llm_provider,
            "str",
            "llm",
            "LLM provider",
            allowed=["ollama", "openrouter"],
        ),
        _param("llm_model", s.llm_model, "str", "llm", "LLM model name"),
        _param("llm_temperature", str(s.llm_temperature), "float", "llm", "LLM temperature", 0.0, 2.0),
        _param("llm_top_p", str(s.llm_top_p), "float", "llm", "LLM top-p", 0.0, 1.0),
        _param(
            "llm_num_ctx_narrow",
            str(s.llm_num_ctx_narrow),
            "int",
            "llm",
            "LLM context window (narrow)",
            512,
            131072,
        ),
        _param(
            "llm_num_ctx_broad",
            str(s.llm_num_ctx_broad),
            "int",
            "llm",
            "LLM context window (broad)",
            512,
            131072,
        ),
        _param(
            "llm_num_predict_narrow",
            str(s.llm_num_predict_narrow),
            "int",
            "llm",
            "LLM max tokens predict (narrow)",
            64,
            16384,
        ),
        _param(
            "llm_num_predict_broad",
            str(s.llm_num_predict_broad),
            "int",
            "llm",
            "LLM max tokens predict (broad)",
            64,
            16384,
        ),
        # --- OpenRouter ---
        _param("openrouter_model", s.openrouter_model, "str", "openrouter", "OpenRouter model name"),
        # --- ML Provider ---
        _param(
            "ml_provider",
            s.ml_provider,
            "str",
            "ml",
            "ML provider for embedding/reranking",
            allowed=["tei", "deepinfra"],
        ),
        _param(
            "deepinfra_embed_model",
            s.deepinfra_embed_model,
            "str",
            "ml",
            "DeepInfra embedding model",
        ),
        _param(
            "deepinfra_rerank_model",
            s.deepinfra_rerank_model,
            "str",
            "ml",
            "DeepInfra reranker model",
        ),
        # --- OCR ---
        _param("ocr_enabled", json.dumps(s.ocr_enabled), "bool", "ocr", "Enable OCR processing"),
        _param("ocr_engine", s.ocr_engine, "str", "ocr", "OCR engine", allowed=["paddleocr", "surya"]),
        _param("ocr_dpi", str(s.ocr_dpi), "int", "ocr", "OCR DPI for scanned pages", 72, 600),
        _param(
            "ocr_min_chars", str(s.ocr_min_chars), "int", "ocr", "Minimum chars to consider page text", 0, 500
        ),
        _param("ocr_lang_paddle", s.ocr_lang_paddle, "str", "ocr", "PaddleOCR language"),
        # --- Storage ---
        _param("s3_endpoint", s.s3_endpoint, "str", "storage", "S3 endpoint URL"),
        _param("s3_bucket", s.s3_bucket, "str", "storage", "S3 bucket name"),
        _param("s3_region", s.s3_region, "str", "storage", "S3 region"),
    ]


async def initialize_app(uow_factory) -> None:
    """Run all startup initialization: bootstrap admin + seed config + load config from DB."""
    await _bootstrap_admin(uow_factory)
    await _seed_config_defaults(uow_factory)
    await _load_config_from_db(uow_factory)


async def _bootstrap_admin(uow_factory) -> None:
    """Ensure default admin user exists."""
    try:
        if not settings.admin_email or not settings.admin_password:
            logger.warning(
                "No admin exists and ADMIN_EMAIL/ADMIN_PASSWORD not set — "
                "you won't be able to log in. Set them in server/.env and restart."
            )
            return

        hasher = BCryptPasswordHasher()
        hashed = await hasher.hash(settings.admin_password)
        async with uow_factory.create(master=True) as uow:
            await uow.users.ensure_admin(
                email=settings.admin_email,
                hashed_password=hashed,
                role=UserRole.ADMIN,
                kind=UserKind.INTERNAL,
            )
            logger.info("Admin ensured: %s", settings.admin_email)
    except Exception as e:
        logger.warning("Failed to bootstrap admin: %s", e)


async def _seed_config_defaults(uow_factory) -> None:
    """Seed config_parameters table with defaults from env.

    Uses upsert: existing parameters keep their DB values,
    missing parameters are added from env defaults.
    Category is always synced to the current definition.
    """
    try:
        async with uow_factory.create(master=True) as uow:
            existing = await uow.config_parameters.get_all()
            existing_keys = {p.key for p in existing}
            existing_categories = {p.key: p.category for p in existing}

            defaults = _build_defaults()
            added = 0
            updated = 0
            for entity in defaults:
                if entity.key not in existing_keys:
                    await uow.config_parameters.save(entity)
                    added += 1
                elif existing_categories.get(entity.key) != entity.category:
                    await uow.config_parameters.update_category(entity.key, entity.category)
                    updated += 1

            if added or updated:
                logger.info(
                    "Config parameters: added %d, category fixed %d (total: %d)",
                    added,
                    updated,
                    len(defaults),
                )
            else:
                logger.info("All %d config parameters already present", len(defaults))
    except Exception as e:
        logger.warning("Failed to seed config defaults: %s", e)


async def _load_config_from_db(uow_factory) -> None:
    """При старте — прогнать все сохранённые параметры через событийную шину.

    Единый путь применения конфига: и runtime-обновления, и startup идут
    via ConfigParameterChanged → EventBus → подписчики.
    """
    try:
        async with uow_factory.create(master=True) as uow:
            rows = await uow.config_parameters.get_all()
            for r in rows:
                event_bus.publish(
                    ConfigParameterChanged(
                        key=r.key,
                        old_value=None,
                        new_value=r.value.strip('"').strip("'"),
                        value_type=r.value_type,
                    )
                )
            logger.info("Loaded %d config parameters via event bus", len(rows))
    except Exception as e:
        logger.warning("Failed to load config from DB: %s", e)
