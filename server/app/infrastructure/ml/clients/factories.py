"""Pure creation functions for ML clients — no cache, no state.

Every call creates a fresh instance.  Caching and lifecycle management
is handled by ``MLClientRegistry`` (infrastructure.ml.client_registry).
"""

from __future__ import annotations

import logging


import httpx
from config import Settings, settings
from domain.value_objects.llm_provider import Breadth, LLMProvider
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from infrastructure.ml.clients.deepinfra_clients import DeepInfraEmbeddingsClient, DeepInfraRerankerClient
from infrastructure.ml.clients.tei_clients import TEIEmbeddingsClient, TEIRerankerClient
from infrastructure.ml.config.embedding_identity import configured_embedding_identity
from qdrant_client import QdrantClient

log = logging.getLogger("default")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def create_embeddings(model: Settings | None = None):
    model = (model or settings).model_copy()
    pool_limits = httpx.Limits(
        max_connections=model.embedding_http_max_connections,
        max_keepalive_connections=model.embedding_http_max_keepalive_connections,
    )
    if model.ml_provider == "deepinfra":
        log.info("Creating DeepInfra embeddings client (model=%s) ...", model.deepinfra_embed_model)
        client = DeepInfraEmbeddingsClient(
            api_key=model.deepinfra_api_key,
            base_url=f"{model.deepinfra_base_url}/openai",
            model=model.deepinfra_embed_model,
            pool_limits=pool_limits,
        )

        client.embedding_identity = configured_embedding_identity(model)
        return client

    log.info("Creating TEI embeddings client (%s) ...", model.tei_embed_url)

    tei_client = TEIEmbeddingsClient(model.tei_embed_url, pool_limits=pool_limits)
    tei_client.embedding_identity = configured_embedding_identity(model)
    return tei_client


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


def create_reranker(model: Settings | None = None):
    model = (model or settings).model_copy()
    pool_limits = httpx.Limits(
        max_connections=model.rerank_http_max_connections,
        max_keepalive_connections=model.rerank_http_max_keepalive_connections,
    )
    if model.ml_provider == "deepinfra":
        log.info("Creating DeepInfra reranker client (model=%s) ...", model.deepinfra_rerank_model)
        return DeepInfraRerankerClient(
            api_key=model.deepinfra_api_key,
            base_url=model.deepinfra_base_url,
            model=model.deepinfra_rerank_model,
            pool_limits=pool_limits,
        )

    log.info("Creating TEI reranker client (%s) ...", model.tei_rerank_url)
    return TEIRerankerClient(model.tei_rerank_url, pool_limits=pool_limits)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def create_llm(
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    model: Settings | None = None,
):
    """Create LLM based on configured provider (ollama or openrouter)."""
    model = (model or settings).model_copy()
    if model.llm_provider == LLMProvider.OPENROUTER:
        return create_openrouter_llm(model=model, temperature=temperature, max_tokens=num_predict)
    return create_ollama_llm(model=model, temperature=temperature, top_p=top_p, num_ctx=num_ctx)


def create_llm_for_breadth(
    breadth: str,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx_narrow: int | None = None,
    num_ctx_broad: int | None = None,
    num_predict_narrow: int | None = None,
    num_predict_broad: int | None = None,
    model: Settings | None = None,
):
    """Create LLM with parameters matching breadth mode."""
    model = (model or settings).model_copy()
    if model.llm_provider == LLMProvider.OPENROUTER:
        return create_openrouter_llm_for_breadth(
            breadth,
            model=model,
            temperature=temperature,
            max_tokens_narrow=num_predict_narrow,
            max_tokens_broad=num_predict_broad,
        )
    return create_ollama_llm_for_breadth(
        breadth,
        model=model,
        temperature=temperature,
        top_p=top_p,
        num_ctx_narrow=num_ctx_narrow,
        num_ctx_broad=num_ctx_broad,
        num_predict_narrow=num_predict_narrow,
        num_predict_broad=num_predict_broad,
    )


def create_ollama_llm(
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    model: Settings | None = None,
) -> ChatOllama:
    model = (model or settings).model_copy()
    return ChatOllama(
        model=model.llm_model,
        base_url=model.ollama_base_url,
        temperature=temperature if temperature is not None else model.llm_temperature,
        top_p=top_p if top_p is not None else model.llm_top_p,
        num_ctx=num_ctx if num_ctx is not None else model.llm_num_ctx_narrow,
        client_kwargs={"timeout": model.llm_request_timeout},
    )


def create_openrouter_llm(
    temperature: float | None = None, max_tokens: int | None = None, model: Settings | None = None
) -> ChatOpenAI:
    model = (model or settings).model_copy()
    return ChatOpenAI(
        model_name=model.openrouter_model,
        openai_api_key=SecretStr(model.openrouter_api_key) if model.openrouter_api_key else None,
        openai_api_base=model.openrouter_base_url,
        temperature=temperature if temperature is not None else model.llm_temperature,
        max_tokens=max_tokens if max_tokens is not None else model.llm_num_predict_narrow,
        request_timeout=120,
        max_retries=2,
        stream_usage=True,
        http_client=httpx.Client(follow_redirects=False),
        http_async_client=httpx.AsyncClient(follow_redirects=False),
    )


def create_ollama_llm_for_breadth(
    breadth: str,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx_narrow: int | None = None,
    num_ctx_broad: int | None = None,
    num_predict_narrow: int | None = None,
    num_predict_broad: int | None = None,
    model: Settings | None = None,
) -> ChatOllama:
    model = (model or settings).model_copy()
    is_broad = breadth == Breadth.BROAD
    num_predict = (num_predict_broad if is_broad else num_predict_narrow) or (
        model.llm_num_predict_broad if is_broad else model.llm_num_predict_narrow
    )
    num_ctx = (num_ctx_broad if is_broad else num_ctx_narrow) or (
        model.llm_num_ctx_broad if is_broad else model.llm_num_ctx_narrow
    )
    return ChatOllama(
        model=model.llm_model,
        base_url=model.ollama_base_url,
        temperature=temperature if temperature is not None else model.llm_temperature,
        top_p=top_p if top_p is not None else model.llm_top_p,
        num_predict=num_predict,
        num_ctx=num_ctx,
        client_kwargs={"timeout": model.llm_request_timeout},
    )


def create_openrouter_llm_for_breadth(
    breadth: str,
    temperature: float | None = None,
    max_tokens_narrow: int | None = None,
    max_tokens_broad: int | None = None,
    model: Settings | None = None,
) -> ChatOpenAI:
    model = (model or settings).model_copy()
    is_broad = breadth == Breadth.BROAD
    max_tokens = (max_tokens_broad if is_broad else max_tokens_narrow) or (
        model.llm_num_predict_broad if is_broad else model.llm_num_predict_narrow
    )
    return ChatOpenAI(
        model_name=model.openrouter_model,
        openai_api_key=SecretStr(model.openrouter_api_key) if model.openrouter_api_key else None,
        openai_api_base=model.openrouter_base_url,
        temperature=temperature if temperature is not None else model.llm_temperature,
        max_tokens=max_tokens,
        request_timeout=120,
        max_retries=2,
        stream_usage=True,
        http_client=httpx.Client(follow_redirects=False),
        http_async_client=httpx.AsyncClient(follow_redirects=False),
    )


# ---------------------------------------------------------------------------
# Fast LLM for auxiliary calls (condense, relevance, decomposition)
# ---------------------------------------------------------------------------


def create_fast_llm_for_auxiliary(model: Settings | None = None):
    """Create a fast, lightweight LLM for auxiliary pipeline calls.

    Uses a smaller model to reduce latency on condense/relevance/decomposition
    calls, leaving the main generation model unchanged.
    """
    model = (model or settings).model_copy()
    if model.llm_provider == LLMProvider.OPENROUTER:
        return ChatOpenAI(
            model_name="meta-llama/llama-3.1-8b-instruct",
            openai_api_key=SecretStr(model.openrouter_api_key) if model.openrouter_api_key else None,
            openai_api_base=model.openrouter_base_url,
            temperature=0,
            max_tokens=200,
            request_timeout=10,
            max_retries=1,
            http_client=httpx.Client(follow_redirects=False),
            http_async_client=httpx.AsyncClient(follow_redirects=False),
        )
    return create_ollama_llm(model=model)


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
