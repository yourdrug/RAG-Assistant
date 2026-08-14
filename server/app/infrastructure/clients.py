"""Lazy-loaded ML and infrastructure clients via functools.lru_cache.

Provides module-level singleton accessors for the embedding model, LLM,
reranker, Qdrant vector store, BM25 index, and Ollama chat clients.
No globals, no classes, no DI container -- each getter returns a cached
instance created on first call.

NOTE: These are process-local caches by design.  Each server/worker
instance must load its own copy of the model weights into memory (models
cannot be shared between processes via Redis).  Cache invalidation on
config changes is handled separately via Postgres LISTEN/NOTIFY
(``infrastructure/events/postgres_config_listener.py``) plus periodic
resync (``Scheduler._periodic_config_resync``), which broadcasts to all
instances simultaneously.  Do NOT replace these with Redis-backed caches.
"""

import functools
import logging
from pathlib import Path

import httpx
from config import settings
from domain.value_objects.llm_provider import Breadth, LLMProvider
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

log = logging.getLogger("default")


@functools.lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    log.info("Loading embedding model %s on %s ...", settings.embed_model, settings.embed_resolved_device)
    return HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        model_kwargs={"device": settings.embed_resolved_device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": settings.embed_batch_size},
    )


@functools.lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.collection_name,
    )


@functools.lru_cache(maxsize=1)
def get_llm():
    """Return LLM based on configured provider (ollama or openrouter)."""
    if settings.llm_provider == LLMProvider.OPENROUTER:
        return _get_openrouter_llm()
    return _get_ollama_llm()


def _get_ollama_llm() -> ChatOllama:
    """Create Ollama LLM instance."""
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        num_ctx=settings.llm_num_ctx_narrow,
    )


def _get_openrouter_llm() -> ChatOpenAI:
    """Create OpenRouter LLM instance (OpenAI-compatible API)."""
    return ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_num_predict_narrow,
    )


def get_llm_for_breadth(breadth: str):
    """Return LLM with parameters matching breadth mode."""
    if settings.llm_provider == LLMProvider.OPENROUTER:
        return _get_openrouter_llm_for_breadth(breadth)
    return _get_ollama_llm_for_breadth(breadth)


def _get_ollama_llm_for_breadth(breadth: str) -> ChatOllama:
    """Return Ollama LLM with num_predict and num_ctx matching breadth mode."""
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


def _get_openrouter_llm_for_breadth(breadth: str) -> ChatOpenAI:
    """Return OpenRouter LLM with max_tokens matching breadth mode."""
    max_tokens = (
        settings.llm_num_predict_broad if breadth == Breadth.BROAD else settings.llm_num_predict_narrow
    )
    return ChatOpenAI(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=settings.llm_temperature,
        max_tokens=max_tokens,
    )


@functools.lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    log.info("Loading reranker %s on %s ...", settings.rerank_model, settings.rerank_resolved_device)
    reranker = CrossEncoder(
        settings.rerank_model,
        max_length=1024,
        device=settings.rerank_resolved_device,
    )
    log.info("Reranker loaded")
    return reranker


@functools.lru_cache(maxsize=1)
def get_qdrant_client():
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


@functools.lru_cache(maxsize=1)
def get_bm25_index():
    """Lazy-load BM25 index from disk. Returns None if not found."""
    from infrastructure.ml.hybrid import load_bm25_index  # nested to avoid circular import

    bm25_path = Path(settings.data_dir) / "bm25_index.json"
    index = load_bm25_index(bm25_path)
    if index is None:
        log.info("No BM25 index found at %s — hybrid search disabled for this run", bm25_path)
    return index


async def get_openrouter_models() -> list[dict]:
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
                # Skip embedding/vision-only models
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

            # Sort by name for easier selection
            models.sort(key=lambda x: x["name"].lower())
            return models
    except Exception as e:
        log.warning("Failed to fetch OpenRouter models: %s", e)
        return []
