"""Prepare a coherent ML client generation before publishing its settings."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import logging
import math
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit


if TYPE_CHECKING:
    from collections.abc import Iterator

    from application.ports.ml_clients import EmbeddingClientPort, RerankerClientPort
    from config import Settings
    from infrastructure.ml.clients.client_registry import MLClientRegistry

import httpx

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
        "llm_base_url",
        "llm_model",
        "llm_auxiliary_model",
        "llm_temperature",
        "llm_top_p",
        "llm_disable_reasoning",
        "llm_num_ctx_narrow",
        "llm_num_ctx_broad",
        "llm_num_predict_narrow",
        "llm_num_predict_broad",
    }
)
EMBEDDING_CLIENT_KEYS = frozenset(
    {
        "embedding_base_url",
        "embedding_http_max_connections",
        "embedding_http_max_keepalive_connections",
    }
)
RERANK_ENDPOINT_KEYS = frozenset({"rerank_base_url", "rerank_model", "rerank_api_format"})
RERANK_CLIENT_KEYS = RERANK_ENDPOINT_KEYS | frozenset(
    {
        "rerank_http_max_connections",
        "rerank_http_max_keepalive_connections",
    }
)


# Optional discovery routes (``/models``, TEI ``/info``) that a server may not implement.
MODELS_LISTING_UNSUPPORTED = frozenset(
    {httpx.codes.NOT_FOUND, httpx.codes.METHOD_NOT_ALLOWED, httpx.codes.NOT_IMPLEMENTED}
)


async def dispose_clients(clients: dict[str, object]) -> None:
    values: list[object] = []

    for value in clients.values():
        values.extend(value.values() if isinstance(value, dict) else [value])

    # Cleanup remains owned even if the resync was cancelled during a probe.
    async def close_all() -> None:
        await asyncio.gather(*(cast("ManagedLLM", value).close() for value in values if value is not None))

    await finish_owned_task(asyncio.create_task(close_all()))


async def serving_model_info(base_url: str) -> dict[str, Any] | None:
    """TEI-style ``/info`` at the endpoint origin, or ``None`` when the server has none.

    Only TEI reports immutable weights identity (``model_sha``); other OpenAI-compatible
    servers are verified by model name and vector dimension alone. No key is sent here.
    """
    parsed = urlsplit(base_url)
    async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
        response = await client.get(f"{parsed.scheme}://{parsed.netloc}/info")

    if response.status_code in MODELS_LISTING_UNSUPPORTED:
        return None

    response.raise_for_status()

    try:
        info = response.json()
    except ValueError:
        return None

    return info if isinstance(info, dict) and info.get("model_id") else None


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
        prepared["cached_llm"] = ManagedLLM(factories.create_llm(model=candidate))

    if registry.cached_fast_llm is not None:
        prepared["cached_fast_llm"] = ManagedLLM(factories.create_fast_llm_for_auxiliary(model=candidate))

    for breadth in registry.cached_llm_breadth_cache:
        cast("dict[str, ManagedLLM]", prepared["cached_llm_breadth_cache"])[breadth] = ManagedLLM(
            factories.create_llm_for_breadth(breadth, model=candidate)
        )

    if registry.cached_instructor is not None:
        prepared["cached_instructor"] = ManagedInstructor(
            create_async_instructor_client(
                candidate.llm_base_url,
                candidate.llm_api_key.get_secret_value(),
            )
        )


LLM_ENDPOINT_KEYS = frozenset({"llm_base_url"})
# The endpoint serves a fixed model list and context window: re-check them on these changes too.
LLM_CAPACITY_KEYS = frozenset(
    {
        "llm_model",
        "llm_auxiliary_model",
        "llm_num_ctx_narrow",
        "llm_num_ctx_broad",
    }
)


def llm_capacity_errors(candidate: Settings, payload: object) -> list[str]:
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

    for model in dict.fromkeys((candidate.llm_model, candidate.active_auxiliary_model)):
        if model not in served:
            errors.append(f"LLM endpoint does not serve model {model!r}")
            continue

        window = served[model].get("max_model_len")

        if isinstance(window, int) and window < budget:
            errors.append(
                f"Model {model!r} context window {window} is smaller than the RAG context budget "
                f"{budget} (llm_num_ctx_narrow/llm_num_ctx_broad); lower the budget or raise "
                "VLLM_MAX_MODEL_LEN"
            )

    return errors


async def probe_llm_endpoint(candidate: Settings, keys: set[str], prepared: dict[str, object]) -> None:
    if not keys & (LLM_ENDPOINT_KEYS | LLM_CAPACITY_KEYS):
        return

    if prepared["cached_llm"] is None:
        prepared["cached_llm"] = ManagedLLM(factories.create_llm(model=candidate))

    # The origin is already approved by endpoint_policy; this probe only checks reachability,
    # credentials and the served model list.
    async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
        response = await client.get(
            f"{candidate.llm_base_url.rstrip('/')}/models",
            headers=llm_auth_headers(candidate.llm_api_key),
        )

    if response.status_code in MODELS_LISTING_UNSUPPORTED:
        log.warning(
            "LLM endpoint does not list models (HTTP %s); model names are not verified",
            response.status_code,
        )
        return

    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError:
        log.warning("LLM endpoint returned a non-JSON model list; model names are not verified")
        return

    errors = llm_capacity_errors(candidate, payload)

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
    if "embedding_base_url" not in keys:
        return

    info = await serving_model_info(candidate.embedding_base_url)

    if info is not None:
        if info["model_id"] != candidate.embedding_model:
            message = "Embedding endpoint serves a different model"
            raise ValidationError(message)

        previous = (
            await serving_model_info(current.embedding_base_url)
            if applied or registry.cached_embeddings is not None
            else None
        )

        if previous is not None:
            # A local path and a null sha cannot prove immutable weights identity.
            if not previous.get("model_sha") or not info.get("model_sha"):
                message = "Embedding endpoint must expose a model revision before endpoint switching"
                raise ValidationError(message)

            identity_fields = ("model_id", "model_sha", "model_dtype", "model_type")

            if any(info.get(key) != previous.get(key) for key in identity_fields):
                message = "Embedding identity changed; use the corpus migration workflow"
                raise ValidationError(message)

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
    try:
        if keys & LLM_CLIENT_KEYS:
            prepare_llm_clients(registry, candidate, prepared)
            with endpoint_availability(component="LLM", allow_unavailable=allow_unavailable):
                await probe_llm_endpoint(candidate, keys, prepared)

        if keys & EMBEDDING_CLIENT_KEYS:
            prepared["cached_embeddings"] = factories.create_embeddings(candidate)
            with endpoint_availability(component="Embedding", allow_unavailable=allow_unavailable):
                await probe_embedding_endpoint(registry, current, candidate, keys, prepared, applied=applied)

        if keys & RERANK_CLIENT_KEYS:
            prepared["cached_reranker"] = factories.create_reranker(candidate)

            if keys & RERANK_ENDPOINT_KEYS:
                with endpoint_availability(component="Reranker", allow_unavailable=allow_unavailable):
                    await cast("RerankerClientPort", prepared["cached_reranker"]).predict(
                        [("compatibility probe", "compatibility probe")]
                    )

    except BaseException:
        await dispose_clients(prepared)
        raise
    else:
        return prepared
