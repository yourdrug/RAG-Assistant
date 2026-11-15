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
from pydantic import SecretStr

from application.runtime.enums import MLProvider
from config import LLMSection, VectorStoreSection, ml_client_settings, process_section
from domain.rag.breadth import Breadth
from infrastructure.corpus import snapshot as corpus_snapshot
from infrastructure.corpus.clients import raw_corpus_client
from infrastructure.corpus.locks import corpus_write_guard
from infrastructure.corpus.snapshot import BoundQdrantClient, CorpusSnapshot
from infrastructure.database.database import database
from infrastructure.ml.clients.deepinfra_clients import DeepInfraEmbeddingsClient, DeepInfraRerankerClient
from infrastructure.ml.clients.instructor_client import NO_API_KEY, llm_auth_headers
from infrastructure.ml.clients.tei_clients import TEIEmbeddingsClient, TEIRerankerClient
from infrastructure.ml.config.embedding_identity import configured_embedding_identity


if TYPE_CHECKING:
    from config import MLClientSettings

log = logging.getLogger("default")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def create_embeddings(
    model: MLClientSettings | None = None,
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


def create_reranker(model: MLClientSettings | None = None) -> DeepInfraRerankerClient | TEIRerankerClient:
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
        )

    log.info("Creating TEI reranker client (%s) ...", model.tei_rerank_url)
    return TEIRerankerClient(model.tei_rerank_url, pool_limits=pool_limits)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


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


class BreadthModelOptions(TypedDict, total=False):
    model: MLClientSettings | None
    temperature: float | None
    top_p: float | None
    num_ctx_narrow: int | None
    num_ctx_broad: int | None
    num_predict_narrow: int | None
    num_predict_broad: int | None


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


def build_chat_model(role: ChatModelRole, **options: Unpack[ChatModelOptions]) -> ChatOpenAI:
    """Build a chat client from one settings snapshot and its call role."""
    model = (options.get("model") or ml_client_settings()).model_copy()
    temperature = options.get("temperature")
    num_predict = options.get("num_predict")
    model_name = options.get("model_name")
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
    default_name = model.active_auxiliary_model if role == ChatModelRole.AUXILIARY else model.llm_model
    # Keyless servers (vLLM, Ollama) still need a non-empty key for the OpenAI SDK.
    api_key = model.llm_api_key if model.llm_api_key.get_secret_value() else SecretStr(NO_API_KEY)

    parameters: OpenAIParameters = {
        "model_name": model_name if model_name is not None else default_name,
        "openai_api_key": api_key,
        "openai_api_base": model.llm_base_url,
        "temperature": temperature if temperature is not None else role_temperature,
        "max_tokens": num_predict if num_predict is not None else output_limits[role],
        "max_retries": 0,
        # Endpoint origins are restricted to deployment-approved lists by endpoint_policy.
        "http_client": httpx.Client(follow_redirects=False),
        "http_async_client": httpx.AsyncClient(follow_redirects=False),
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
) -> ChatOpenAI:
    """Create LLM for the configured OpenAI-compatible endpoint."""
    return build_chat_model(
        ChatModelRole.DEFAULT,
        model=model,
        temperature=temperature,
        top_p=top_p,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )


def create_llm_for_breadth(breadth: str, **options: Unpack[BreadthModelOptions]) -> ChatOpenAI:
    """Create LLM with parameters matching breadth mode."""
    model = options.get("model")
    temperature = options.get("temperature")
    top_p = options.get("top_p")
    num_ctx_broad = options.get("num_ctx_broad")
    num_ctx_narrow = options.get("num_ctx_narrow")
    num_predict_broad = options.get("num_predict_broad")
    num_predict_narrow = options.get("num_predict_narrow")
    is_broad = breadth == Breadth.BROAD
    return build_chat_model(
        ChatModelRole.BROAD if is_broad else ChatModelRole.NARROW,
        model=model,
        temperature=temperature,
        top_p=top_p,
        num_ctx=(num_ctx_broad if is_broad else num_ctx_narrow) or None,
        num_predict=(num_predict_broad if is_broad else num_predict_narrow) or None,
    )


def create_fast_llm_for_auxiliary(model: MLClientSettings | None = None) -> ChatOpenAI:
    """Create the lightweight client for condense, relevance and decomposition."""
    return build_chat_model(ChatModelRole.AUXILIARY, model=model)


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
# LLM endpoint models (async, not cached — called rarely)
# ---------------------------------------------------------------------------

NON_CHAT_MODEL_MARKERS = ("embedding", "vision", "tts", "whisper")


def model_price_per_million(pricing: object, field: str) -> float | None:
    raw = pricing.get(field) if isinstance(pricing, dict) else None

    if not isinstance(raw, (str, int, float)) or raw == "":
        return None

    try:
        return round(float(raw) * 1_000_000, 4)
    except ValueError:
        return None


def normalize_llm_model(item: object) -> dict[str, Any] | None:
    """One OpenAI ``/models`` entry; optional OpenRouter/vLLM metadata is kept when present."""
    if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
        return None

    model_id = item["id"]

    if any(marker in model_id.lower() for marker in NON_CHAT_MODEL_MARKERS):
        return None

    window = item.get("context_length") or item.get("max_model_len")
    pricing = item.get("pricing")
    prompt = model_price_per_million(pricing, "prompt")
    completion = model_price_per_million(pricing, "completion")
    return {
        "id": model_id,
        "name": item.get("name") if isinstance(item.get("name"), str) and item.get("name") else model_id,
        "context_length": window if isinstance(window, int) else None,
        "pricing": {"prompt": prompt, "completion": completion}
        if prompt is not None and completion is not None
        else None,
    }


async def fetch_llm_models() -> list[dict[str, Any]]:
    """List chat models served by the configured endpoint via the standard ``GET /models``.

    Returns an empty list when the endpoint does not implement it; the admin UI then
    falls back to free-text model names.
    """
    llm_settings = process_section(LLMSection)

    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
            response = await client.get(
                f"{llm_settings.llm_base_url.rstrip('/')}/models",
                headers=llm_auth_headers(llm_settings.llm_api_key),
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Failed to fetch LLM endpoint models: %s", type(exc).__name__)
        return []

    data = payload.get("data") if isinstance(payload, dict) else None
    models = [model for model in map(normalize_llm_model, data if isinstance(data, list) else []) if model]
    models.sort(key=lambda model: model["name"].lower())
    return models
