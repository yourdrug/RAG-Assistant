"""Own synchronous and asynchronous benchmark judge HTTP clients."""

import asyncio
import threading

import httpx
import instructor
from openai import AsyncOpenAI, OpenAI

from application.runtime.enums import LLMProvider
from config import BenchmarkSection, LLMSection, process_section
from infrastructure.benchmark.judging.token_usage import record_judge_usage
from infrastructure.budget.runtime import BudgetRuntime
from infrastructure.budget.transport import PaidAsyncTransport, PaidSyncTransport
from infrastructure.ml.clients.factories import FAST_OPENROUTER_MODEL
from infrastructure.ml.clients.instructor_client import NO_API_KEY
from infrastructure.ml.clients.managed_sync_instructor import ManagedSyncInstructor


_judge_client_cache: dict[tuple[BudgetRuntime | None, str, str, str], ManagedSyncInstructor] = {}


_judge_client_lock = threading.Lock()


RETIRED_JUDGE_CLIENTS: set[ManagedSyncInstructor] = set()


def get_judge_model() -> str:
    """Resolve the judge model: explicit setting > provider's fast/auxiliary model."""
    benchmark_settings = process_section(BenchmarkSection)

    if benchmark_settings.benchmark_judge_model:
        return benchmark_settings.benchmark_judge_model

    llm_settings = process_section(LLMSection)

    if llm_settings.llm_provider == LLMProvider.OPENAI_COMPATIBLE.value:
        return llm_settings.active_auxiliary_model

    # OpenRouter: use fast/cheap model for judge calls
    return FAST_OPENROUTER_MODEL


def judge_endpoint(llm_settings: LLMSection) -> tuple[str, str, bool]:
    """Judge calls follow the active LLM provider so a local setup never leaks data to OpenRouter.

    Returns ``(base_url, api_key, metered)``; only the OpenRouter endpoint is budget-metered.
    """
    metered = llm_settings.llm_provider == LLMProvider.OPENROUTER.value
    api_key = llm_settings.active_llm_api_key.get_secret_value() or NO_API_KEY
    return llm_settings.active_llm_base_url, api_key, metered


def get_judge_client(model: str, *, runtime: BudgetRuntime | None = None) -> ManagedSyncInstructor:
    """Get or create a cached instructor client for the judge model.

    Thread-safe: first call creates the client, subsequent calls return the cached
    instance.  This avoids creating a new httpx connection pool per benchmark question.
    """
    snapshot = process_section(LLMSection).model_copy()
    base_url, api_key, metered = judge_endpoint(snapshot)
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
            api_key=api_key,
            timeout=snapshot.llm_auxiliary_timeout,
            max_retries=0,
            http_client=httpx.Client(
                follow_redirects=False, transport=PaidSyncTransport(runtime=runtime) if metered else None
            ),
        )
        mode = instructor.Mode.JSON
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
    llm_settings = process_section(LLMSection)
    base_url, api_key, metered = judge_endpoint(llm_settings)

    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=llm_settings.llm_auxiliary_timeout,
        max_retries=0,
        http_client=httpx.AsyncClient(
            follow_redirects=False, transport=PaidAsyncTransport(runtime=runtime) if metered else None
        ),
    )
