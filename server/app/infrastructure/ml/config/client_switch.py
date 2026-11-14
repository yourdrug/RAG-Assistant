"""Prepare a coherent ML client generation before publishing its settings."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import logging
import math
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from collections.abc import Iterator

    from application.ports.ml_clients import EmbeddingClientPort, RerankerClientPort
    from config import Settings
    from infrastructure.ml.clients.client_registry import MLClientRegistry

import httpx

from application.budget import billing_principal
from application.runtime.enums import LLMProvider, MLProvider
from domain.shared.exceptions import ValidationError
from infrastructure.ml.clients import factories
from infrastructure.ml.clients.instructor_client import create_async_instructor_client, llm_auth_headers
from infrastructure.ml.clients.managed_llm import ManagedInstructor, ManagedLLM
from infrastructure.utils.async_tasks import finish_owned_task


log = logging.getLogger("default")


@contextmanager
def endpoint_availability(*, component: str, allow_unavailable: bool) -> Iterator[None]:
    try:
        yield
    except (httpx.TransportError, httpx.HTTPStatusError) as exc:
        if not allow_unavailable or (
            isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code < httpx.codes.INTERNAL_SERVER_ERROR
        ):
            raise

        log.warning(
            "%s endpoint is unavailable during startup (%s); continuing with the saved ML "
            "configuration. ML operations may fail until the endpoint is restored. "
            "Changing embedding providers requires the corpus migration workflow.",
            component,
            type(exc).__name__,
        )


LLM_CLIENT_KEYS = frozenset(
    {
        "openrouter_base_url",
        "llm_provider",
        "openrouter_model",
        "openai_compatible_base_url",
        "openai_compatible_model",
        "openai_compatible_auxiliary_model",
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
        prepared["cached_instructor"] = ManagedInstructor(
            create_async_instructor_client(
                candidate.active_llm_base_url,
                candidate.active_llm_api_key.get_secret_value(),
                runtime=registry.budget_runtime,
            )
        )


LLM_ENDPOINT_KEYS = frozenset({"openrouter_base_url", "openai_compatible_base_url", "llm_provider"})
# Self-hosted servers serve a fixed model list and context window: re-check them on these changes too.
LOCAL_LLM_CAPACITY_KEYS = frozenset(
    {
        "openai_compatible_model",
        "openai_compatible_auxiliary_model",
        "llm_num_ctx_narrow",
        "llm_num_ctx_broad",
    }
)


def local_llm_capacity_errors(candidate: Settings, payload: object) -> list[str]:
    """Check served models and context window reported by an OpenAI-compatible ``/models``.

    vLLM reports ``max_model_len`` per model; servers that omit it (Ollama, LM Studio) are only
    checked for the model id. RAG packs context up to ``llm_num_ctx_*``, so a smaller server window
    would reject long broad queries at request time instead of at configuration time.
    """
    data = payload.get("data") if isinstance(payload, dict) else None

    if not isinstance(data, list):
        return []

    served = {item["id"]: item for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)}
    errors: list[str] = []
    budget = max(candidate.llm_num_ctx_narrow, candidate.llm_num_ctx_broad)

    for model in dict.fromkeys((candidate.openai_compatible_model, candidate.active_auxiliary_model)):
        if model not in served:
            errors.append(f"OpenAI-compatible server does not serve model {model!r}")
            continue

        window = served[model].get("max_model_len")

        if isinstance(window, int) and window < budget:
            errors.append(
                f"Model {model!r} context window {window} is smaller than the RAG context budget "
                f"{budget} (llm_num_ctx_narrow/llm_num_ctx_broad); lower the budget or raise "
                "VLLM_MAX_MODEL_LEN"
            )

    return errors


async def probe_llm_endpoint(
    registry: MLClientRegistry, candidate: Settings, keys: set[str], prepared: dict[str, object]
) -> None:
    local = candidate.llm_provider == LLMProvider.OPENAI_COMPATIBLE.value
    trigger = LLM_ENDPOINT_KEYS | LOCAL_LLM_CAPACITY_KEYS if local else LLM_ENDPOINT_KEYS

    if not keys & trigger:
        return

    if prepared["cached_llm"] is None:
        prepared["cached_llm"] = ManagedLLM(
            factories.create_llm(model=candidate, runtime=registry.budget_runtime)
        )

    async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
        response = await client.get(
            f"{candidate.active_llm_base_url.rstrip('/')}/models",
            headers=llm_auth_headers(candidate.active_llm_api_key),
        )
        response.raise_for_status()

    if local:
        errors = local_llm_capacity_errors(candidate, response.json())

        if errors:
            raise ValidationError("; ".join(errors))


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
    registry: MLClientRegistry,
    current: Settings,
    candidate: Settings,
    keys: set[str],
    *,
    applied: bool,
    allow_unavailable: bool = False,
) -> dict[str, Any]:
    prepared: dict[str, object] = {}
    billing_token = billing_principal.set(billing_principal.get() or "system:configuration-probe")
    try:
        if keys & LLM_CLIENT_KEYS:
            prepare_llm_clients(registry, candidate, prepared)
            with endpoint_availability(component="LLM", allow_unavailable=allow_unavailable):
                await probe_llm_endpoint(registry, candidate, keys, prepared)

        if keys & EMBEDDING_CLIENT_KEYS:
            prepared["cached_embeddings"] = factories.create_embeddings(
                candidate, runtime=registry.budget_runtime
            )
            with endpoint_availability(component="Embedding", allow_unavailable=allow_unavailable):
                await probe_embedding_endpoint(registry, current, candidate, keys, prepared, applied=applied)

        if keys & RERANK_CLIENT_KEYS:
            prepared["cached_reranker"] = factories.create_reranker(
                candidate, runtime=registry.budget_runtime
            )

            if ("tei_rerank_url" in keys and candidate.ml_provider == MLProvider.TEI.value) or (
                "deepinfra_base_url" in keys and candidate.ml_provider == MLProvider.DEEPINFRA.value
            ):
                with endpoint_availability(component="Reranker", allow_unavailable=allow_unavailable):
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
