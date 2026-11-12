"""Pure creation functions for ML clients — no cache, no state.

Every call creates a fresh instance.  Caching and lifecycle management
is handled by ``MLClientRegistry`` (infrastructure.ml.client_registry).
"""

from __future__ import annotations

from enum import StrEnum
from functools import partial
import logging
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

import httpx
from langchain_openai import ChatOpenAI

from application.runtime.enums import LLMProvider, MLProvider
from config import LLMSection, VectorStoreSection, ml_client_settings, process_section
from domain.rag.breadth import Breadth
from infrastructure.budget.transport import PaidAsyncTransport, PaidSyncTransport
from infrastructure.corpus import snapshot as corpus_snapshot
from infrastructure.corpus.clients import raw_corpus_client
from infrastructure.corpus.locks import corpus_write_guard
from infrastructure.corpus.snapshot import BoundQdrantClient, CorpusSnapshot
from infrastructure.database.database import database
from infrastructure.ml.clients.deepinfra_clients import DeepInfraEmbeddingsClient, DeepInfraRerankerClient
from infrastructure.ml.clients.tei_clients import TEIEmbeddingsClient, TEIRerankerClient
from infrastructure.ml.config.embedding_identity import configured_embedding_identity


if TYPE_CHECKING:
    from pydantic import SecretStr

    from config import MLClientSettings
    from infrastructure.budget.runtime import BudgetRuntime

log = logging.getLogger("default")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def create_embeddings(
    model: MLClientSettings | None = None, *, runtime: BudgetRuntime | None = None
) -> DeepInfraEmbeddingsClient | TEIEmbeddingsClient:
    model = (model or ml_client_settings()).model_copy()
    pool_limits = httpx.Limits(
        max_connections=model.embedding_http_max_connections,
        max_keepalive_connections=model.embedding_http_max_keepalive_connections,
    )

    if model.ml_provider == MLProvider.DEEPINFRA.value:
        log.info("Creating DeepInfra embeddings client (model=%s) ...", model.deepinfra_embed_model)
        client = DeepInfraEmbeddingsClient(
            api_key=model.deepinfra_api_key.get_secret_value(),
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


def create_reranker(
    model: MLClientSettings | None = None, *, runtime: BudgetRuntime | None = None
) -> DeepInfraRerankerClient | TEIRerankerClient:
    model = (model or ml_client_settings()).model_copy()
    pool_limits = httpx.Limits(
        max_connections=model.rerank_http_max_connections,
        max_keepalive_connections=model.rerank_http_max_keepalive_connections,
    )

    if model.ml_provider == MLProvider.DEEPINFRA.value:
        log.info("Creating DeepInfra reranker client (model=%s) ...", model.deepinfra_rerank_model)
        return DeepInfraRerankerClient(
            api_key=model.deepinfra_api_key.get_secret_value(),
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


class ChatModelOptions(TypedDict, total=False):
    model: MLClientSettings | None
    temperature: float | None
    top_p: float | None
    num_ctx: int | None
    num_predict: int | None
    model_name: str | None
    base_url: str | None
    runtime: BudgetRuntime | None


class BreadthModelOptions(TypedDict, total=False):
    model: MLClientSettings | None
    temperature: float | None
    top_p: float | None
    num_ctx_narrow: int | None
    num_ctx_broad: int | None
    num_predict_narrow: int | None
    num_predict_broad: int | None
    runtime: BudgetRuntime | None


class OpenAIParameters(TypedDict, total=False):
    model_name: str
    openai_api_key: SecretStr | None
    openai_api_base: str
    temperature: float
    max_tokens: int
    max_retries: int
    http_client: httpx.Client
    http_async_client: httpx.AsyncClient
    request_timeout: float
    stream_usage: bool


def build_chat_model(
    provider: LLMProvider | str,
    role: ChatModelRole,
    **options: Unpack[ChatModelOptions],
) -> ChatOpenAI:
    """Build a chat client from one settings snapshot and its call role."""
    model = (options.get("model") or ml_client_settings()).model_copy()
    temperature = options.get("temperature")
    num_predict = options.get("num_predict")
    model_name = options.get("model_name")
    runtime = options.get("runtime")
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

    if provider != LLMProvider.OPENROUTER:
        message = f"Unsupported LLM provider: {provider}"
        raise ValueError(message)

    default_name = FAST_OPENROUTER_MODEL if role == ChatModelRole.AUXILIARY else model.openrouter_model
    parameters: OpenAIParameters = {
        "model_name": model_name if model_name is not None else default_name,
        "openai_api_key": model.openrouter_api_key if model.openrouter_api_key.get_secret_value() else None,
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


def create_llm(
    temperature: float | None = None,
    top_p: float | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    model: MLClientSettings | None = None,
    *,
    runtime: BudgetRuntime | None = None,
) -> ChatOpenAI:
    """Create LLM based on the configured provider."""
    return build_chat_model(
        (model or ml_client_settings()).llm_provider,
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
    model: MLClientSettings | None = None,
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


def create_llm_for_breadth(breadth: str, **options: Unpack[BreadthModelOptions]) -> ChatOpenAI:
    """Create LLM with parameters matching breadth mode."""
    model = options.get("model")
    runtime = options.get("runtime")
    temperature = options.get("temperature")
    top_p = options.get("top_p")
    num_ctx_broad = options.get("num_ctx_broad")
    num_ctx_narrow = options.get("num_ctx_narrow")
    num_predict_broad = options.get("num_predict_broad")
    num_predict_narrow = options.get("num_predict_narrow")
    is_broad = breadth == Breadth.BROAD
    return build_chat_model(
        (model or ml_client_settings()).llm_provider,
        ChatModelRole.BROAD if is_broad else ChatModelRole.NARROW,
        model=model,
        runtime=runtime,
        temperature=temperature,
        top_p=top_p,
        num_ctx=(num_ctx_broad if is_broad else num_ctx_narrow) or None,
        num_predict=(num_predict_broad if is_broad else num_predict_narrow) or None,
    )


def create_fast_llm_for_auxiliary(
    model: MLClientSettings | None = None, *, runtime: BudgetRuntime | None = None
) -> ChatOpenAI:
    """Create the lightweight client for condense, relevance and decomposition."""
    return build_chat_model(
        (model or ml_client_settings()).llm_provider, ChatModelRole.AUXILIARY, model=model, runtime=runtime
    )


# ---------------------------------------------------------------------------
# Qdrant client
# ---------------------------------------------------------------------------


def create_qdrant_client() -> BoundQdrantClient:
    vector_store_settings = process_section(VectorStoreSection)
    snapshot = corpus_snapshot.installed_corpus or CorpusSnapshot(
        vector_store_settings.qdrant_url,
        vector_store_settings.collection_name,
        vector_store_settings.collection_name,
        configured_embedding_identity(),
    )
    raw = raw_corpus_client(snapshot.endpoint)
    return BoundQdrantClient(raw, snapshot, partial(corpus_write_guard, database))


# ---------------------------------------------------------------------------
# OpenRouter models (async, not cached — called rarely)
# ---------------------------------------------------------------------------


async def fetch_openrouter_models() -> list[dict[str, Any]]:
    """Fetch available models from OpenRouter API.

    Returns list of dicts with keys: id, name, context_length, pricing.
    Filters to chat-capable models only.
    """
    llm_settings = process_section(LLMSection)

    if not llm_settings.openrouter_api_key.get_secret_value():
        return []

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{llm_settings.openrouter_base_url}/models",
                headers={"Authorization": f"Bearer {llm_settings.openrouter_api_key.get_secret_value()}"},
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
        log.warning("Failed to fetch OpenRouter models: %s", e, exc_info=True)
        return []
