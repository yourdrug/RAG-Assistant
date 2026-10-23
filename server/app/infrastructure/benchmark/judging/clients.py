"""Own synchronous and asynchronous benchmark judge HTTP clients."""

from infrastructure.budget.runtime import BudgetRuntime
from openai import AsyncOpenAI
from infrastructure.budget.transport import PaidAsyncTransport

import asyncio
import threading
import instructor
import httpx
from infrastructure.budget.transport import PaidSyncTransport
from openai import OpenAI
from typing import Any
from config import settings
from domain.runtime.enums import LLMProvider
from infrastructure.ml.clients.factories import FAST_OPENROUTER_MODEL
from infrastructure.ml.clients.managed_sync_instructor import ManagedSyncInstructor
from infrastructure.benchmark.judging.token_usage import record_judge_usage


_judge_client_cache: dict[tuple[BudgetRuntime | None, str, str, str], Any] = {}


_judge_client_lock = threading.Lock()


RETIRED_JUDGE_CLIENTS: set[ManagedSyncInstructor] = set()


def get_judge_model() -> str:
    """Resolve the judge model: explicit setting > fast model for OpenRouter > llm_model for Ollama."""
    if settings.benchmark_judge_model:
        return settings.benchmark_judge_model

    if settings.llm_provider == LLMProvider.OLLAMA:
        return settings.llm_model

    # OpenRouter: use fast/cheap model for judge calls
    return FAST_OPENROUTER_MODEL


def get_judge_client(model: str, *, runtime: BudgetRuntime | None = None):
    """Get or create a cached instructor client for the judge model.

    Thread-safe: first call creates the client, subsequent calls return the cached
    instance.  This avoids creating a new httpx connection pool per benchmark question.
    """
    snapshot = settings.model_copy()
    remote = snapshot.llm_provider == LLMProvider.OPENROUTER
    base_url = snapshot.openrouter_base_url if remote else f"{snapshot.ollama_base_url.rstrip('/')}/v1"
    key = (runtime, snapshot.llm_provider, base_url, model)
    with _judge_client_lock:
        RETIRED_JUDGE_CLIENTS.difference_update(
            [client for client in RETIRED_JUDGE_CLIENTS if client.is_closed]
        )
        cached = _judge_client_cache.get(key)

        if cached is not None:
            return cached

        stale = [item for item in _judge_client_cache if item[0] is runtime and item[1:3] != key[1:3]]

        for item in stale:
            retired = _judge_client_cache.pop(item)
            RETIRED_JUDGE_CLIENTS.add(retired)
            retired.retire()

        # OpenRouter's Instructor adapter supports JSON, but rejects MD_JSON.
        # Both modes avoid requiring tool calling from the chosen model.
        raw = OpenAI(
            base_url=base_url,
            api_key=snapshot.openrouter_api_key
            if snapshot.llm_provider == LLMProvider.OPENROUTER
            else "ollama",
            timeout=snapshot.llm_auxiliary_timeout,
            max_retries=0,
            http_client=httpx.Client(
                follow_redirects=False, transport=PaidSyncTransport(runtime=runtime) if remote else None
            ),
        )
        mode = (
            instructor.Mode.JSON
            if snapshot.llm_provider == LLMProvider.OPENROUTER
            else instructor.Mode.MD_JSON
        )
        client = instructor.from_openai(raw, mode=mode)

        client.on("completion:response", record_judge_usage)
        managed = ManagedSyncInstructor(client, budget_runtime=runtime)
        _judge_client_cache[key] = managed
        return managed


def retire_judge_clients(*, runtime: BudgetRuntime | None = None) -> None:
    with _judge_client_lock:
        keys = [key for key in _judge_client_cache if key[0] is runtime]
        clients = [_judge_client_cache.pop(key) for key in keys]
        RETIRED_JUDGE_CLIENTS.difference_update(
            [client for client in RETIRED_JUDGE_CLIENTS if client.is_closed]
        )
        RETIRED_JUDGE_CLIENTS.update(clients)

    for client in clients:
        client.retire()


async def close_judge_clients(*, runtime: BudgetRuntime | None = None) -> None:
    retire_judge_clients(runtime=runtime)
    with _judge_client_lock:
        clients = [client for client in RETIRED_JUDGE_CLIENTS if client.budget_runtime is runtime]
    await asyncio.gather(*(client.close() for client in clients))
    with _judge_client_lock:
        RETIRED_JUDGE_CLIENTS.difference_update(clients)


def create_async_judge_client(*, runtime: BudgetRuntime | None = None) -> AsyncOpenAI:
    remote = settings.llm_provider == LLMProvider.OPENROUTER
    return AsyncOpenAI(
        base_url=settings.openrouter_base_url if remote else f"{settings.ollama_base_url}/v1",
        api_key=settings.openrouter_api_key if remote else "ollama",
        timeout=settings.llm_auxiliary_timeout,
        max_retries=0,
        http_client=httpx.AsyncClient(
            follow_redirects=False, transport=PaidAsyncTransport(runtime=runtime) if remote else None
        ),
    )
