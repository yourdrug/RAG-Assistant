"""Pure creation functions for ML clients — no cache, no state.

Every call creates a fresh instance.  Caching and lifecycle management
is handled by ``MLClientRegistry`` (infrastructure.ml.client_registry).
"""

from __future__ import annotations
from typing import TYPE_CHECKING


import logging
from enum import StrEnum


import httpx
from domain.runtime.enums import MLProvider
from functools import partial
from infrastructure.corpus.locks import corpus_write_guard
from infrastructure.corpus.clients import raw_corpus_client
from infrastructure.database.database import database
from infrastructure.corpus import snapshot as corpus_snapshot
from infrastructure.corpus.snapshot import BoundQdrantClient, CorpusSnapshot
from infrastructure.budget.transport import PaidSyncTransport, PaidAsyncTransport
from config import Settings, settings
from domain.rag.breadth import Breadth
from domain.runtime.enums import LLMProvider
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from infrastructure.ml.clients.deepinfra_clients import DeepInfraEmbeddingsClient, DeepInfraRerankerClient
from infrastructure.ml.clients.tei_clients import TEIEmbeddingsClient, TEIRerankerClient
from infrastructure.ml.config.embedding_identity import configured_embedding_identity

if TYPE_CHECKING:
    from infrastructure.budget.runtime import BudgetRuntime

log = logging.getLogger("default")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def create_embeddings(model: Settings | None = None, *, runtime: BudgetRuntime | None = None):
    model = (model or settings).model_copy()
    pool_limits = httpx.Limits(
        max_connections=model.embedding_http_max_connections,
        max_keepalive_connections=model.embedding_http_max_keepalive_connections,
    )

    if model.ml_provider == MLProvider.DEEPINFRA.value:
        log.info("Creating DeepInfra embeddings client (model=%s) ...", model.deepinfra_embed_model)
        client = DeepInfraEmbeddingsClient(
            api_key=model.deepinfra_api_key,
            base_url=f"{model.deepinfra_base_url}/openai",
            model=model.deepinfra_embed_model,
            pool_limits=pool_limits,
            runtime=runtime,
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


def create_reranker(model: Settings | None = None, *, runtime: BudgetRuntime | None = None):
    model = (model or settings).model_copy()
    pool_limits = httpx.Limits(
        max_connections=model.rerank_http_max_connections,
        max_keepalive_connections=model.rerank_http_max_keepalive_connections,
    )

    if model.ml_provider == MLProvider.DEEPINFRA.value:
        log.info("Creating DeepInfra reranker client (model=%s) ...", model.deepinfra_rerank_model)
        return DeepInfraRerankerClient(
            api_key=model.deepinfra_api_key,
            base_url=model.deepinfra_base_url,
            model=model.deepinfra_rerank_model,
            pool_limits=pool_limits,
            runtime=runtime,
        )

    log.info("Creating TEI reranker client (%s) ...", model.tei_rerank_url)
    return TEIRerankerClient(model.tei_rerank_url, pool_limits=pool_limits)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


FAST_OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"


class ChatModelRole(StrEnum):
    DEFAULT = "default"
    NARROW = "narrow"
    BROAD = "broad"
    AUXILIARY = "auxiliary"
    BENCHMARK = "benchmark"


def build_chat_model(
    provider: LLMProvider | str,
    role: ChatModelRole,
    *,
    model: Settings | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
    runtime: BudgetRuntime | None = None,
) -> ChatOllama | ChatOpenAI:
    """Build a chat client from one settings snapshot and its call role."""
    model = (model or settings).model_copy()
    output_limits = {
        ChatModelRole.DEFAULT: model.llm_num_predict_narrow,
        ChatModelRole.NARROW: model.llm_num_predict_narrow,
        ChatModelRole.BROAD: model.llm_num_predict_broad,
        ChatModelRole.AUXILIARY: 200,
        ChatModelRole.BENCHMARK: model.llm_num_predict_broad,
    }
    role_temperature = (
        0 if role in (ChatModelRole.AUXILIARY, ChatModelRole.BENCHMARK) else model.llm_temperature
    )

    if provider == LLMProvider.OPENROUTER:
        default_name = FAST_OPENROUTER_MODEL if role == ChatModelRole.AUXILIARY else model.openrouter_model
        parameters = {
            "model_name": model_name if model_name is not None else default_name,
            "openai_api_key": SecretStr(model.openrouter_api_key) if model.openrouter_api_key else None,
            "openai_api_base": model.openrouter_base_url,
            "temperature": temperature if temperature is not None else role_temperature,
            "max_tokens": num_predict if num_predict is not None else output_limits[role],
            "max_retries": 0,
            "http_client": httpx.Client(follow_redirects=False, transport=PaidSyncTransport(runtime=runtime)),
            "http_async_client": httpx.AsyncClient(
                follow_redirects=False, transport=PaidAsyncTransport(runtime=runtime)
            ),
        }

        if role != ChatModelRole.BENCHMARK:
            parameters["request_timeout"] = 10 if role == ChatModelRole.AUXILIARY else 120

        if role not in (ChatModelRole.AUXILIARY, ChatModelRole.BENCHMARK):
            parameters["stream_usage"] = True

        return ChatOpenAI(**parameters)

    parameters = {
        "model": model_name if model_name is not None else model.llm_model,
        "base_url": base_url if base_url is not None else model.ollama_base_url,
        "temperature": temperature
        if temperature is not None
        else (0 if role == ChatModelRole.BENCHMARK else model.llm_temperature),
    }

    if role != ChatModelRole.BENCHMARK:
        parameters.update(
            top_p=top_p if top_p is not None else model.llm_top_p,
            num_ctx=num_ctx
            if num_ctx is not None
            else (model.llm_num_ctx_broad if role == ChatModelRole.BROAD else model.llm_num_ctx_narrow),
            client_kwargs={"timeout": model.llm_request_timeout},
        )

    if role in (ChatModelRole.NARROW, ChatModelRole.BROAD):
        parameters["num_predict"] = num_predict if num_predict is not None else output_limits[role]

    return ChatOllama(**parameters)


def create_llm(
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    model: Settings | None = None,
    *,
    runtime: BudgetRuntime | None = None,
):
    """Create LLM based on the configured provider."""
    return build_chat_model(
        (model or settings).llm_provider,
        ChatModelRole.DEFAULT,
        model=model,
        runtime=runtime,
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )


def create_openrouter_llm(
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: Settings | None = None,
    *,
    runtime: BudgetRuntime | None = None,
) -> ChatOpenAI:
    """Create an OpenRouter chat client through the shared provider factory."""
    return build_chat_model(
        LLMProvider.OPENROUTER,
        ChatModelRole.DEFAULT,
        model=model,
        runtime=runtime,
        temperature=temperature,
        num_predict=max_tokens,
    )


def create_llm_for_breadth(
    breadth: str,
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx_narrow: int | None = None,
    num_ctx_broad: int | None = None,
    num_predict_narrow: int | None = None,
    num_predict_broad: int | None = None,
    model: Settings | None = None,
    *,
    runtime: BudgetRuntime | None = None,
):
    """Create LLM with parameters matching breadth mode."""
    is_broad = breadth == Breadth.BROAD
    return build_chat_model(
        (model or settings).llm_provider,
        ChatModelRole.BROAD if is_broad else ChatModelRole.NARROW,
        model=model,
        runtime=runtime,
        temperature=temperature,
        top_p=top_p,
        num_ctx=(num_ctx_broad if is_broad else num_ctx_narrow) or None,
        num_predict=(num_predict_broad if is_broad else num_predict_narrow) or None,
    )


def create_fast_llm_for_auxiliary(model: Settings | None = None, *, runtime: BudgetRuntime | None = None):
    """Create the lightweight client for condense, relevance and decomposition."""
    return build_chat_model(
        (model or settings).llm_provider, ChatModelRole.AUXILIARY, model=model, runtime=runtime
    )


# ---------------------------------------------------------------------------
# Qdrant client
# ---------------------------------------------------------------------------


def create_qdrant_client() -> BoundQdrantClient:
    snapshot = corpus_snapshot.installed_corpus or CorpusSnapshot(
        settings.qdrant_url,
        settings.collection_name,
        settings.collection_name,
        configured_embedding_identity(),
    )
    raw = raw_corpus_client(snapshot.endpoint)
    return BoundQdrantClient(raw, snapshot, partial(corpus_write_guard, database))


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
