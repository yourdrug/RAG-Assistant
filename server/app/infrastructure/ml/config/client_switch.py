"""Prepare a coherent ML client generation before publishing its settings."""

import asyncio
import math
from domain.runtime.enums import MLProvider
from typing import Any

import httpx

from config import Settings
from application.budget import billing_principal
from domain.shared.exceptions import ValidationError
from domain.runtime.enums import LLMProvider
from infrastructure.ml.clients import factories
from infrastructure.ml.clients.instructor_client import create_async_instructor_client
from infrastructure.ml.clients.managed_llm import ManagedInstructor, ManagedLLM
from infrastructure.storage.file_storage import finish_owned_task

LLM_CLIENT_KEYS = frozenset(
    {
        "ollama_base_url",
        "openrouter_base_url",
        "llm_provider",
        "llm_model",
        "openrouter_model",
        "llm_temperature",
        "llm_top_p",
        "llm_num_ctx_narrow",
        "llm_num_ctx_broad",
        "llm_num_predict_narrow",
        "llm_num_predict_broad",
    }
)
EMBEDDING_CLIENT_KEYS = frozenset(
    {
        "ml_provider",
        "deepinfra_embed_model",
        "tei_embed_url",
        "deepinfra_base_url",
        "embedding_http_max_connections",
        "embedding_http_max_keepalive_connections",
    }
)
RERANK_CLIENT_KEYS = frozenset(
    {
        "ml_provider",
        "deepinfra_rerank_model",
        "tei_rerank_url",
        "deepinfra_base_url",
        "rerank_http_max_connections",
        "rerank_http_max_keepalive_connections",
    }
)


async def dispose_clients(clients: dict) -> None:
    values: list[Any] = []
    for value in clients.values():
        values.extend(value.values() if isinstance(value, dict) else [value])

    # Cleanup remains owned even if the resync was cancelled during a probe.
    async def close_all():
        await asyncio.gather(*(value.close() for value in values if value is not None))

    await finish_owned_task(asyncio.create_task(close_all()))


async def tei_model_info(url: str) -> dict:
    async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
        response = await client.get(f"{url.rstrip('/')}/info")
        response.raise_for_status()
        info = response.json()
    if not isinstance(info, dict) or not info.get("model_id"):
        raise ValidationError("TEI embedding identity is unavailable")
    return info


def check_embedding_vector(vector: list[float], dimension: int) -> None:
    if (
        not isinstance(vector, list)
        or len(vector) != dimension
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector)
    ):
        raise ValidationError("Embedding endpoint returned an incompatible vector")


def prepare_llm_clients(registry, candidate: Settings, prepared: dict) -> None:
    # Include empty slots: a lazy old client can appear while a network probe yields.
    prepared.update({"_llm": None, "_fast_llm": None, "_instructor": None, "_llm_breadth_cache": {}})
    if registry._llm is not None:
        prepared["_llm"] = ManagedLLM(factories.create_llm(model=candidate))
    if registry._fast_llm is not None:
        prepared["_fast_llm"] = ManagedLLM(factories.create_fast_llm_for_auxiliary(model=candidate))
    for breadth in registry._llm_breadth_cache:
        prepared["_llm_breadth_cache"][breadth] = ManagedLLM(
            factories.create_llm_for_breadth(breadth, model=candidate)
        )
    if registry._instructor is not None:
        remote = candidate.llm_provider == LLMProvider.OPENROUTER
        prepared["_instructor"] = ManagedInstructor(
            create_async_instructor_client(
                candidate.openrouter_base_url if remote else f"{candidate.ollama_base_url.rstrip('/')}/v1",
                candidate.openrouter_api_key if remote else LLMProvider.OLLAMA.value,
            )
        )


async def probe_llm_endpoint(candidate: Settings, keys: set[str], prepared: dict) -> None:
    remote = candidate.llm_provider == LLMProvider.OPENROUTER
    endpoint = "openrouter_base_url" if remote else "ollama_base_url"
    if endpoint not in keys:
        return
    if prepared["_llm"] is None:
        prepared["_llm"] = ManagedLLM(factories.create_llm(model=candidate))
    async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
        response = await client.get(
            f"{getattr(candidate, endpoint).rstrip('/')}/{'models' if remote else 'api/tags'}",
            headers={"Authorization": f"Bearer {candidate.openrouter_api_key}"} if remote else {},
        )
        response.raise_for_status()


async def probe_embedding_endpoint(registry, current, candidate, keys, prepared, applied) -> None:
    if "tei_embed_url" in keys and candidate.ml_provider == MLProvider.TEI.value:
        info = await tei_model_info(candidate.tei_embed_url)
        if info["model_id"] != candidate.tei_embed_model:
            raise ValidationError("TEI serves a different embedding model")
        if applied or registry._embeddings is not None:
            previous = await tei_model_info(current.tei_embed_url)
            # A local path and a null sha cannot prove immutable weights identity.
            if not previous.get("model_sha") or not info.get("model_sha"):
                raise ValidationError("TEI must expose a model revision before endpoint switching")
            identity_fields = ("model_id", "model_sha", "model_dtype", "model_type")
            if any(info.get(key) != previous.get(key) for key in identity_fields):
                raise ValidationError("TEI embedding identity changed; use the corpus migration workflow")
        check_embedding_vector(
            await prepared["_embeddings"].embed_query("embedding endpoint compatibility probe"),
            candidate.embed_dim,
        )
    if "deepinfra_base_url" in keys and candidate.ml_provider == MLProvider.DEEPINFRA.value:
        check_embedding_vector(
            await prepared["_embeddings"].embed_query("embedding endpoint compatibility probe"),
            candidate.embed_dim,
        )


async def prepare_clients(
    registry, current: Settings, candidate: Settings, keys: set[str], *, applied: bool
) -> dict:
    prepared: dict[str, Any] = {}
    billing_token = billing_principal.set(billing_principal.get() or 'system:configuration-probe')
    try:
        if keys & LLM_CLIENT_KEYS:
            prepare_llm_clients(registry, candidate, prepared)
            await probe_llm_endpoint(candidate, keys, prepared)
        if keys & EMBEDDING_CLIENT_KEYS:
            prepared["_embeddings"] = factories.create_embeddings(candidate)
            await probe_embedding_endpoint(registry, current, candidate, keys, prepared, applied)
        if keys & RERANK_CLIENT_KEYS:
            prepared["_reranker"] = factories.create_reranker(candidate)
            if (
                "tei_rerank_url" in keys
                and candidate.ml_provider == MLProvider.TEI.value
                or "deepinfra_base_url" in keys
                and candidate.ml_provider == MLProvider.DEEPINFRA.value
            ):
                await prepared["_reranker"].predict([("compatibility probe", "compatibility probe")])
        return prepared
    except BaseException:
        await dispose_clients(prepared)
        raise
    finally:
        billing_principal.reset(billing_token)
