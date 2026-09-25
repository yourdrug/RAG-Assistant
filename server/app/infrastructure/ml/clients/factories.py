"""Pure creation functions for ML clients — no cache, no state.

Every call creates a fresh instance.  Caching and lifecycle management
is handled by ``MLClientRegistry`` (infrastructure.ml.client_registry).
"""

from __future__ import annotations

import logging

import httpx
from config import settings
from domain.value_objects.llm_provider import Breadth, LLMProvider
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from qdrant_client import QdrantClient

log = logging.getLogger("default")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def create_embeddings():
    pool_limits = httpx.Limits(
        max_connections=settings.http_pool_max_connections,
        max_keepalive_connections=settings.http_pool_max_keepalive,
    )
    if settings.ml_provider == "deepinfra":
        from infrastructure.ml.clients.deepinfra_clients import DeepInfraEmbeddingsClient

        log.info("Creating DeepInfra embeddings client (model=%s) ...", settings.deepinfra_embed_model)
        return DeepInfraEmbeddingsClient(
            api_key=settings.deepinfra_api_key,
            base_url=f"{settings.deepinfra_base_url}/openai",
            model=settings.deepinfra_embed_model,
            pool_limits=pool_limits,
        )
    from infrastructure.ml.clients.tei_clients import TEIEmbeddingsClient

    log.info("Creating TEI embeddings client (%s) ...", settings.tei_embed_url)
    return TEIEmbeddingsClient(settings.tei_embed_url, pool_limits=pool_limits)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


def create_reranker():
    pool_limits = httpx.Limits(
        max_connections=settings.http_pool_max_connections // 2,
        max_keepalive_connections=settings.http_pool_max_keepalive // 2,
    )
    if settings.ml_provider == "deepinfra":
        from infrastructure.ml.clients.deepinfra_clients import DeepInfraRerankerClient

        log.info("Creating DeepInfra reranker client (model=%s) ...", settings.deepinfra_rerank_model)
        return DeepInfraRerankerClient(
            api_key=settings.deepinfra_api_key,
            base_url=settings.deepinfra_base_url,
            model=settings.deepinfra_rerank_model,
            pool_limits=pool_limits,
        )
    from infrastructure.ml.clients.tei_clients import TEIRerankerClient

    log.info("Creating TEI reranker client (%s) ...", settings.tei_rerank_url)
    return TEIRerankerClient(settings.tei_rerank_url, pool_limits=pool_limits)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def create_llm(
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
):
    """Create LLM based on configured provider (ollama or openrouter)."""
    if settings.llm_provider == LLMProvider.OPENROUTER:
        return _create_openrouter_llm(temperature=temperature, max_tokens=num_predict)
    return _create_ollama_llm(temperature=temperature, top_p=top_p, num_ctx=num_ctx)


def create_llm_for_breadth(
    breadth: str,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx_narrow: int | None = None,
    num_ctx_broad: int | None = None,
    num_predict_narrow: int | None = None,
    num_predict_broad: int | None = None,
):
    """Create LLM with parameters matching breadth mode."""
    if settings.llm_provider == LLMProvider.OPENROUTER:
        return _create_openrouter_llm_for_breadth(
            breadth,
            temperature=temperature,
            max_tokens_narrow=num_predict_narrow,
            max_tokens_broad=num_predict_broad,
        )
    return _create_ollama_llm_for_breadth(
        breadth,
        temperature=temperature,
        top_p=top_p,
        num_ctx_narrow=num_ctx_narrow,
        num_ctx_broad=num_ctx_broad,
        num_predict_narrow=num_predict_narrow,
        num_predict_broad=num_predict_broad,
    )


def _create_ollama_llm(
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
) -> ChatOllama:
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        top_p=top_p if top_p is not None else settings.llm_top_p,
        num_ctx=num_ctx if num_ctx is not None else settings.llm_num_ctx_narrow,
        client_kwargs={"timeout": settings.llm_request_timeout},
    )


def _create_openrouter_llm(temperature: float | None = None, max_tokens: int | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model_name=settings.openrouter_model,
        openai_api_key=SecretStr(settings.openrouter_api_key) if settings.openrouter_api_key else None,
        openai_api_base=settings.openrouter_base_url,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.llm_num_predict_narrow,
        request_timeout=120,
        max_retries=2,
        stream_usage=True,
    )


def _create_ollama_llm_for_breadth(
    breadth: str,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx_narrow: int | None = None,
    num_ctx_broad: int | None = None,
    num_predict_narrow: int | None = None,
    num_predict_broad: int | None = None,
) -> ChatOllama:
    is_broad = breadth == Breadth.BROAD
    num_predict = (num_predict_broad if is_broad else num_predict_narrow) or (
        settings.llm_num_predict_broad if is_broad else settings.llm_num_predict_narrow
    )
    num_ctx = (num_ctx_broad if is_broad else num_ctx_narrow) or (
        settings.llm_num_ctx_broad if is_broad else settings.llm_num_ctx_narrow
    )
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        top_p=top_p if top_p is not None else settings.llm_top_p,
        num_predict=num_predict,
        num_ctx=num_ctx,
        client_kwargs={"timeout": settings.llm_request_timeout},
    )


def _create_openrouter_llm_for_breadth(
    breadth: str,
    temperature: float | None = None,
    max_tokens_narrow: int | None = None,
    max_tokens_broad: int | None = None,
) -> ChatOpenAI:
    is_broad = breadth == Breadth.BROAD
    max_tokens = (max_tokens_broad if is_broad else max_tokens_narrow) or (
        settings.llm_num_predict_broad if is_broad else settings.llm_num_predict_narrow
    )
    return ChatOpenAI(
        model_name=settings.openrouter_model,
        openai_api_key=SecretStr(settings.openrouter_api_key) if settings.openrouter_api_key else None,
        openai_api_base=settings.openrouter_base_url,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens,
        request_timeout=120,
        max_retries=2,
        stream_usage=True,
    )


# ---------------------------------------------------------------------------
# Fast LLM for auxiliary calls (condense, relevance, decomposition)
# ---------------------------------------------------------------------------


def create_fast_llm_for_auxiliary():
    """Create a fast, lightweight LLM for auxiliary pipeline calls.

    Uses a smaller model to reduce latency on condense/relevance/decomposition
    calls, leaving the main generation model unchanged.
    """
    if settings.llm_provider == LLMProvider.OPENROUTER:
        return ChatOpenAI(
            model_name="meta-llama/llama-3.1-8b-instruct",
            openai_api_key=SecretStr(settings.openrouter_api_key) if settings.openrouter_api_key else None,
            openai_api_base=settings.openrouter_base_url,
            temperature=0,
            max_tokens=200,
            request_timeout=10,
            max_retries=1,
        )
    return _create_ollama_llm()


# ---------------------------------------------------------------------------
# Qdrant client
# ---------------------------------------------------------------------------


def create_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout,
    )


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------


def load_bm25_index():
    """Load BM25 index from S3. Returns None if not found."""
    from infrastructure.bm25.persistence import load_bm25_index_from_s3_sync
    from infrastructure.storage import get_storage

    storage = get_storage()
    if not hasattr(storage, "download_bytes"):
        log.warning("BM25 index loading requires S3 storage (no download_bytes support)")
        return None

    index = load_bm25_index_from_s3_sync(storage)
    if index is None:
        log.info("No BM25 index found in S3 — hybrid search disabled for this run")
    return index


# ---------------------------------------------------------------------------
# OpenRouter models (async, not cached — called rarely)
# ---------------------------------------------------------------------------


async def fetch_openrouter_models() -> list[dict]:
    """Fetch available models from OpenRouter API.

    Returns list of dicts with keys: id, name, context_length, pricing.
    Filters to chat-capable models only.
    """
    if not settings.openrouter_api_key:
        return []

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{settings.openrouter_base_url}/models",
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            )
            r.raise_for_status()
            data = r.json()

            models = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                if any(skip in model_id.lower() for skip in ["embedding", "vision", "tts", "whisper"]):
                    continue

                pricing = m.get("pricing", {})
                prompt_price = float(pricing.get("prompt", "0") or "0") * 1_000_000
                completion_price = float(pricing.get("completion", "0") or "0") * 1_000_000

                models.append(
                    {
                        "id": model_id,
                        "name": m.get("name", model_id),
                        "context_length": m.get("context_length", 0),
                        "pricing": {
                            "prompt": round(prompt_price, 4),
                            "completion": round(completion_price, 4),
                        },
                    }
                )

            models.sort(key=lambda x: x["name"].lower())
            return models
    except Exception as e:
        log.warning("Failed to fetch OpenRouter models: %s", e)
        return []
