"""Prepare a coherent ML client generation before publishing its settings."""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from application.ports.ml_clients import EmbeddingClientPort, RerankerClientPort
    from config import Settings
    from infrastructure.ml.clients.client_registry import MLClientRegistry

import httpx

from application.budget import billing_principal
from application.runtime.enums import LLMProvider, MLProvider
from domain.shared.exceptions import ValidationError
from infrastructure.ml.clients import factories
from infrastructure.ml.clients.instructor_client import create_async_instructor_client
from infrastructure.ml.clients.managed_llm import ManagedInstructor, ManagedLLM
from infrastructure.utils.async_tasks import finish_owned_task


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


async def dispose_clients(clients: dict[str, object]) -> None:
    values: list[object] = []

    for value in clients.values():
        values.extend(value.values() if isinstance(value, dict) else [value])

    # Cleanup remains owned even if the resync was cancelled during a probe.
    async def close_all() -> None:
        await asyncio.gather(*(cast("ManagedLLM", value).close() for value in values if value is not None))

    await finish_owned_task(asyncio.create_task(close_all()))


async def tei_model_info(url: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
        response = await client.get(f"{url.rstrip('/')}/info")
        response.raise_for_status()
        info = response.json()

    if not isinstance(info, dict) or not info.get("model_id"):
        message = "TEI embedding identity is unavailable"
        raise ValidationError(message)

    return info


def check_embedding_vector(vector: object, dimension: int) -> None:
    if (
        not isinstance(vector, list)
        or len(vector) != dimension
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector)
    ):
        message = "Embedding endpoint returned an incompatible vector"
        raise ValidationError(message)


def prepare_llm_clients(registry: MLClientRegistry, candidate: Settings, prepared: dict[str, object]) -> None:
    # Include empty slots: a lazy old client can appear while a network probe yields.
    prepared.update(
        {
            "cached_llm": None,
            "cached_fast_llm": None,
            "cached_instructor": None,
            "cached_llm_breadth_cache": {},
        }
    )

    if registry.cached_llm is not None:
        prepared["cached_llm"] = ManagedLLM(
            factories.create_llm(model=candidate, runtime=registry.budget_runtime)
        )

    if registry.cached_fast_llm is not None:
        prepared["cached_fast_llm"] = ManagedLLM(
            factories.create_fast_llm_for_auxiliary(model=candidate, runtime=registry.budget_runtime)
        )

    for breadth in registry.cached_llm_breadth_cache:
        cast("dict[str, ManagedLLM]", prepared["cached_llm_breadth_cache"])[breadth] = ManagedLLM(
            factories.create_llm_for_breadth(breadth, model=candidate, runtime=registry.budget_runtime)
        )

    if registry.cached_instructor is not None:
        remote = candidate.llm_provider == LLMProvider.OPENROUTER
        prepared["cached_instructor"] = ManagedInstructor(
            create_async_instructor_client(
                candidate.openrouter_base_url if remote else f"{candidate.ollama_base_url.rstrip('/')}/v1",
                candidate.openrouter_api_key if remote else LLMProvider.OLLAMA.value,
                runtime=registry.budget_runtime,
            )
        )


async def probe_llm_endpoint(
    registry: MLClientRegistry, candidate: Settings, keys: set[str], prepared: dict[str, object]
) -> None:
    remote = candidate.llm_provider == LLMProvider.OPENROUTER
    endpoint = "openrouter_base_url" if remote else "ollama_base_url"

    if endpoint not in keys:
        return

    if prepared["cached_llm"] is None:
        prepared["cached_llm"] = ManagedLLM(
            factories.create_llm(model=candidate, runtime=registry.budget_runtime)
        )

    async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
        response = await client.get(
            f"{getattr(candidate, endpoint).rstrip('/')}/{'models' if remote else 'api/tags'}",
            headers={"Authorization": f"Bearer {candidate.openrouter_api_key}"} if remote else {},
        )
        response.raise_for_status()


async def probe_embedding_endpoint(
    registry: MLClientRegistry,
    current: Settings,
    candidate: Settings,
    keys: set[str],
    prepared: dict[str, object],
    *,
    applied: bool,
) -> None:
    if "tei_embed_url" in keys and candidate.ml_provider == MLProvider.TEI.value:
        info = await tei_model_info(candidate.tei_embed_url)

        if info["model_id"] != candidate.tei_embed_model:
            message = "TEI serves a different embedding model"
            raise ValidationError(message)

        if applied or registry.cached_embeddings is not None:
            previous = await tei_model_info(current.tei_embed_url)

            # A local path and a null sha cannot prove immutable weights identity.
            if not previous.get("model_sha") or not info.get("model_sha"):
                message = "TEI must expose a model revision before endpoint switching"
                raise ValidationError(message)

            identity_fields = ("model_id", "model_sha", "model_dtype", "model_type")

            if any(info.get(key) != previous.get(key) for key in identity_fields):
                message = "TEI embedding identity changed; use the corpus migration workflow"
                raise ValidationError(message)

        check_embedding_vector(
            await cast("EmbeddingClientPort", prepared["cached_embeddings"]).embed_query(
                "embedding endpoint compatibility probe"
            ),
            candidate.embed_dim,
        )

    if "deepinfra_base_url" in keys and candidate.ml_provider == MLProvider.DEEPINFRA.value:
        check_embedding_vector(
            await cast("EmbeddingClientPort", prepared["cached_embeddings"]).embed_query(
                "embedding endpoint compatibility probe"
            ),
            candidate.embed_dim,
        )


async def prepare_clients(
    registry: MLClientRegistry, current: Settings, candidate: Settings, keys: set[str], *, applied: bool
) -> dict[str, Any]:
    prepared: dict[str, object] = {}
    billing_token = billing_principal.set(billing_principal.get() or "system:configuration-probe")
    try:
        if keys & LLM_CLIENT_KEYS:
            prepare_llm_clients(registry, candidate, prepared)
            await probe_llm_endpoint(registry, candidate, keys, prepared)

        if keys & EMBEDDING_CLIENT_KEYS:
            prepared["cached_embeddings"] = factories.create_embeddings(
                candidate, runtime=registry.budget_runtime
            )
            await probe_embedding_endpoint(registry, current, candidate, keys, prepared, applied=applied)

        if keys & RERANK_CLIENT_KEYS:
            prepared["cached_reranker"] = factories.create_reranker(
                candidate, runtime=registry.budget_runtime
            )

            if ("tei_rerank_url" in keys and candidate.ml_provider == MLProvider.TEI.value) or (
                "deepinfra_base_url" in keys and candidate.ml_provider == MLProvider.DEEPINFRA.value
            ):
                await cast("RerankerClientPort", prepared["cached_reranker"]).predict(
                    [("compatibility probe", "compatibility probe")]
                )

    except BaseException:
        await dispose_clients(prepared)
        raise
    else:
        return prepared
    finally:
        billing_principal.reset(billing_token)
