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
    from infrastructure.ml.tei_clients import TEIEmbeddingsClient

    log.info("Creating TEI embeddings client (%s) ...", settings.tei_embed_url)
    return TEIEmbeddingsClient(settings.tei_embed_url)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


def create_reranker():
    from infrastructure.ml.tei_clients import TEIRerankerClient

    log.info("Creating TEI reranker client (%s) ...", settings.tei_rerank_url)
    return TEIRerankerClient(settings.tei_rerank_url)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def create_llm():
    """Create LLM based on configured provider (ollama or openrouter)."""
    if settings.llm_provider == LLMProvider.OPENROUTER:
        return _create_openrouter_llm()
    return _create_ollama_llm()


def create_llm_for_breadth(breadth: str):
    """Create LLM with parameters matching breadth mode."""
    if settings.llm_provider == LLMProvider.OPENROUTER:
        return _create_openrouter_llm_for_breadth(breadth)
    return _create_ollama_llm_for_breadth(breadth)


def _create_ollama_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        num_ctx=settings.llm_num_ctx_narrow,
    )


def _create_openrouter_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model_name=settings.openrouter_model,
        openai_api_key=SecretStr(settings.openrouter_api_key) if settings.openrouter_api_key else None,
        openai_api_base=settings.openrouter_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_num_predict_narrow,
        request_timeout=120,
        max_retries=2,
        stream_usage=True,
    )


def _create_ollama_llm_for_breadth(breadth: str) -> ChatOllama:
    num_predict = (
        settings.llm_num_predict_broad if breadth == Breadth.BROAD else settings.llm_num_predict_narrow
    )
    num_ctx = settings.llm_num_ctx_broad if breadth == Breadth.BROAD else settings.llm_num_ctx_narrow
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        num_predict=num_predict,
        num_ctx=num_ctx,
    )


def _create_openrouter_llm_for_breadth(breadth: str) -> ChatOpenAI:
    max_tokens = (
        settings.llm_num_predict_broad if breadth == Breadth.BROAD else settings.llm_num_predict_narrow
    )
    return ChatOpenAI(
        model_name=settings.openrouter_model,
        openai_api_key=SecretStr(settings.openrouter_api_key) if settings.openrouter_api_key else None,
        openai_api_base=settings.openrouter_base_url,
        temperature=settings.llm_temperature,
        max_tokens=max_tokens,
        request_timeout=120,
        max_retries=2,
        stream_usage=True,
    )


# ---------------------------------------------------------------------------
# Qdrant client
# ---------------------------------------------------------------------------


def create_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------


def load_bm25_index():
    """Load BM25 index from S3. Returns None if not found."""
    from infrastructure.ml.hybrid import load_bm25_index_from_s3_sync
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
