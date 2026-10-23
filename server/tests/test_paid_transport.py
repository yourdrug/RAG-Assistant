"""Accounting belongs to physical dispatch, before SDK validation."""

from infrastructure.benchmark.judging import clients as judge_clients

from infrastructure.benchmark.judging import requests as judge_requests

import asyncio
import json
from decimal import Decimal
from unittest.mock import AsyncMock

import httpx
import pytest
from instructor.core.exceptions import InstructorRetryException
from infrastructure.ml.clients import factories, managed_http, deepinfra_clients
from infrastructure.ml.clients.managed_llm import ManagedLLM
from domain.runtime.enums import LLMProvider
from infrastructure.benchmark.judging import judge

from application.budget import BudgetPolicy, CostBudgetService, billing_principal
from domain.runtime.cost import BudgetError, CostRate, ModelBillingContract
from infrastructure.budget.runtime import BudgetRuntime
from pydantic import BaseModel
from infrastructure.budget.provider_catalog import OpenRouterBillingCatalog
from infrastructure.budget.transport import PaidAsyncTransport, PaidSyncTransport
from infrastructure.ml.clients import instructor_client
from infrastructure.budget.job_context import bill_job_to_initiator
from infrastructure.ml.config.client_switch import prepare_clients
from infrastructure.ml.clients.client_registry import MLClientRegistry
from config import settings


@pytest.mark.asyncio
async def test_disabled_quota_does_not_require_reservation():
    ledger = AsyncMock()
    ledger.reserve.return_value = True
    async with CostBudgetService(ledger).operation(
        policy=BudgetPolicy(False),
        principal=None,
        operation_id='op',
        provider='openrouter',
        model='m',
        operation='llm',
        maximum_units=None,
        rate_version=1,
    ):
        pass
    ledger.reserve.assert_not_awaited()


def test_embedding_zero_output_budget():
    rate = CostRate(
        'deepinfra', 'BAAI/bge-m3', 1, 'tokens', Decimal('1'), Decimal('0'), 8192, input_only=True
    )
    assert rate.reserve(8192, 0) == 8_192_000


@pytest.fixture
def runtime():
    result = BudgetRuntime()
    result.service = CostBudgetService(AsyncMock())
    result.service.ledger.reserve.return_value = True
    result.catalog = AsyncMock()
    result.catalog.contract.return_value = ModelBillingContract(
        'tokens',
        'llm',
        100,
        frozenset({'model', 'messages', 'max_tokens'}),
        'verified',
    )
    result.policy_snapshot = AsyncMock(return_value=BudgetPolicy(True))
    result.rate = AsyncMock(
        return_value=CostRate(
            'openrouter',
            'm',
            1,
            'tokens',
            Decimal('1'),
            Decimal('2'),
            100,
        )
    )
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize('sync', [False, True])
@pytest.mark.parametrize('provider', ['openrouter', 'deepinfra'])
async def test_disabled_budget_preserves_request_without_backend_or_tariff(runtime, sync, provider):
    runtime.loop = asyncio.get_running_loop()
    runtime.policy_snapshot.return_value = BudgetPolicy(False)
    runtime.service = None
    runtime.catalog = None
    runtime.rate.side_effect = BudgetError('No tariff')
    body = (
        {'model': 'm', 'messages': [{'role': 'user', 'content': 'hello'}]}
        if provider == 'openrouter'
        else {'queries': ['q'], 'documents': ['d']}
    )
    url = (
        'https://openrouter.ai/api/v1/chat/completions'
        if provider == 'openrouter'
        else 'https://api.deepinfra.com/v1/inference/BAAI/bge-reranker-v2-m3'
    )
    sent = []

    def send(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={'result': 'ok'})

    def invoke_sync():
        with httpx.Client(
            transport=PaidSyncTransport(runtime=runtime, transport=httpx.MockTransport(send))
        ) as client:
            return client.post(url, json=body)

    token = billing_principal.set(None)
    try:
        if sync:
            response = await asyncio.to_thread(invoke_sync)
        else:
            async with httpx.AsyncClient(
                transport=PaidAsyncTransport(runtime=runtime, transport=httpx.MockTransport(send))
            ) as client:
                response = await client.post(url, json=body)
        assert response.json() == {'result': 'ok'}
        assert sent == [body]
    finally:
        billing_principal.reset(token)


def test_disabled_sync_budget_does_not_require_an_owner_event_loop(monkeypatch):
    monkeypatch.setattr(settings, 'cost_rate_limit_enabled', False)
    runtime = BudgetRuntime()
    with httpx.Client(
        transport=PaidSyncTransport(
            runtime=runtime,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'result': 'ok'})),
        )
    ) as client:
        response = client.post(
            'https://openrouter.ai/api/v1/chat/completions',
            json={'model': 'm', 'messages': []},
        )
    assert response.json() == {'result': 'ok'}


@pytest.mark.asyncio
@pytest.mark.parametrize('sync', [False, True])
async def test_unknown_budget_policy_never_dispatches(runtime, sync):
    runtime.loop = asyncio.get_running_loop()
    runtime.policy_snapshot.side_effect = ConnectionError('Policy backend unavailable')
    sent = []

    def send(request):
        sent.append(request)
        return httpx.Response(200)

    def invoke_sync():
        with httpx.Client(
            transport=PaidSyncTransport(runtime=runtime, transport=httpx.MockTransport(send))
        ) as client:
            client.post('https://openrouter.ai/api/v1/chat/completions', json={'model': 'm'})

    with pytest.raises(ConnectionError, match='Policy backend'):
        if sync:
            await asyncio.to_thread(invoke_sync)
        else:
            async with httpx.AsyncClient(
                transport=PaidAsyncTransport(runtime=runtime, transport=httpx.MockTransport(send))
            ) as client:
                await client.post('https://openrouter.ai/api/v1/chat/completions', json={'model': 'm'})
    assert not sent


@pytest.mark.asyncio
@pytest.mark.parametrize('principal', [None, 'user:7'])
@pytest.mark.parametrize('failure', [False, True])
async def test_deepinfra_configuration_probe_has_billing_scope(runtime, monkeypatch, principal, failure):
    runtime.catalog = OpenRouterBillingCatalog()
    runtime.rate.return_value = CostRate(
        'deepinfra', 'BAAI/bge-m3', 1, 'tokens', Decimal('1'), Decimal('0'), 32768, input_only=True
    )
    observed = []

    async def reserve(**kwargs):
        observed.append(kwargs['principal'])
        return True

    runtime.service.ledger.reserve.side_effect = reserve

    def send(request):
        if failure:
            raise httpx.ReadTimeout('Probe failed')
        if request.url.path.endswith('/embeddings'):
            return httpx.Response(200, json={'data': [{'embedding': [0.1, 0.2]}]})
        return httpx.Response(200, json={'scores': [0.9]})

    monkeypatch.setattr(
        managed_http,
        'PaidAsyncTransport',
        lambda **kwargs: PaidAsyncTransport(runtime=runtime, transport=httpx.MockTransport(send)),
    )
    model = settings.model_copy(
        update={
            'ml_provider': 'deepinfra',
            'deepinfra_api_key': 'test',
            'embed_dim': 2,
            'deepinfra_embed_model': 'BAAI/bge-m3',
            'deepinfra_rerank_model': 'Qwen/Qwen3-Reranker-4B',
        }
    )
    registry = MLClientRegistry()
    token = billing_principal.set(principal)
    try:
        if failure:
            with pytest.raises(httpx.ReadTimeout):
                await prepare_clients(registry, model, model, {'deepinfra_base_url'}, applied=True)
        else:
            prepared = await prepare_clients(registry, model, model, {'deepinfra_base_url'}, applied=True)
            registry.install_clients(prepared)
        assert observed == [principal or 'system:configuration-probe'] * (1 if failure else 2)
        assert billing_principal.get() == principal
    finally:
        billing_principal.reset(token)
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('valid_usage', [True, False])
async def test_response_settled_before_structured_validation(runtime, valid_usage):
    async def provider(request):
        body = json.loads(request.content)
        assert body['provider']['require_parameters'] is True
        return httpx.Response(
            200,
            json={
                'id': 'gen-1',
                'choices': [{'message': {'content': 'invalid JSON'}}],
                'usage': {'prompt_tokens': 4, 'completion_tokens': 2} if valid_usage else {},
            },
        )

    token = billing_principal.set('user:1')
    try:
        async with httpx.AsyncClient(
            transport=PaidAsyncTransport(
                runtime=runtime,
                transport=httpx.MockTransport(provider),
            )
        ) as client:
            for _ in range(2):
                await client.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    json={
                        'model': 'm',
                        'max_tokens': 10,
                        'messages': [{'role': 'user', 'content': 'hi'}],
                    },
                )
        ledger = runtime.service.ledger
        assert ledger.reserve.await_count == 2
        assert ledger.reconcile.await_count == (2 if valid_usage else 0)
        if valid_usage:
            assert ledger.reconcile.await_args.args[2] == 8000
        assert ledger.remember_response.await_count == 2
    finally:
        billing_principal.reset(token)


@pytest.mark.asyncio
async def test_unsupported_shape_never_reaches_transport(runtime):
    provider = AsyncMock()
    token = billing_principal.set('user:1')
    try:
        async with httpx.AsyncClient(
            transport=PaidAsyncTransport(
                runtime=runtime,
                transport=httpx.MockTransport(provider),
            )
        ) as client:
            with pytest.raises(BudgetError):
                await client.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    json={
                        'model': 'm',
                        'max_tokens': 10,
                        'plugins': [{'id': 'web'}],
                    },
                )
        provider.assert_not_called()
    finally:
        billing_principal.reset(token)


@pytest.mark.asyncio
async def test_timeout_after_dispatch_stays_pending(runtime):
    async def provider(request):
        raise httpx.ReadTimeout('unknown outcome')

    token = billing_principal.set('user:1')
    try:
        async with httpx.AsyncClient(
            transport=PaidAsyncTransport(
                runtime=runtime,
                transport=httpx.MockTransport(provider),
            )
        ) as client:
            with pytest.raises(httpx.ReadTimeout):
                await client.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    json={
                        'model': 'm',
                        'max_tokens': 10,
                        'messages': [],
                    },
                )
        runtime.service.ledger.reconcile.assert_not_called()
    finally:
        billing_principal.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize('operation,count', [('embedding', 1), ('embedding', 3), ('reranking', 2)])
async def test_deepinfra_batch_uses_documented_units(runtime, operation, count):
    runtime.catalog.contract.return_value = ModelBillingContract(
        'tokens',
        operation,
        100,
        frozenset(
            {'model', 'input', 'encoding_format'} if operation == 'embedding' else {'queries', 'documents'}
        ),
        'verified',
    )
    runtime.rate.return_value = CostRate(
        'deepinfra', 'm', 1, 'tokens', Decimal('1'), Decimal('0'), 100, input_only=True
    )
    payload = (
        {'usage': {'prompt_tokens': 7}}
        if operation == 'embedding'
        else {
            'input_tokens': 7,
            'inference_status': {'tokens_input': 999},
            'request_id': 'rerank-id',
        }
    )
    token = billing_principal.set('user:1')
    try:
        async with httpx.AsyncClient(
            transport=PaidAsyncTransport(
                runtime=runtime,
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
            )
        ) as client:
            await client.post(
                'https://api.deepinfra.com/v1/openai/embeddings'
                if operation == 'embedding'
                else 'https://api.deepinfra.com/v1/inference/m',
                json={'model': 'm', 'input': ['x'] * count, 'encoding_format': 'float'}
                if operation == 'embedding'
                else {'queries': ['q'], 'documents': ['d'] * count},
            )
        ledger = runtime.service.ledger
        assert ledger.reserve.await_args.kwargs['maximum_units'] == count * 100_000
        assert ledger.reconcile.await_args.args[2] == 7000
        assert ledger.reconcile.await_args.kwargs['billable_units']['input_tokens'] == 7
    finally:
        billing_principal.reset(token)


@pytest.mark.asyncio
async def test_cancel_before_dispatch_refunds_zero(runtime):
    runtime.service.ledger.mark_dispatch.side_effect = asyncio.CancelledError
    provider = AsyncMock()
    token = billing_principal.set('user:1')
    try:
        async with httpx.AsyncClient(
            transport=PaidAsyncTransport(
                runtime=runtime,
                transport=httpx.MockTransport(provider),
            )
        ) as client:
            with pytest.raises(asyncio.CancelledError):
                await client.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    json={
                        'model': 'm',
                        'max_tokens': 10,
                        'messages': [],
                    },
                )
        provider.assert_not_called()
        assert runtime.service.ledger.reconcile.await_args.args[2] == 0
    finally:
        billing_principal.reset(token)


@pytest.mark.asyncio
async def test_sync_async_use_identical_ledger(runtime):
    runtime.loop = asyncio.get_running_loop()
    payload = {'id': 'gen-1', 'usage': {'cost': '0.00002', 'prompt_tokens': 4, 'completion_tokens': 2}}
    token = billing_principal.set('user:1')
    body = {'model': 'm', 'max_tokens': 10, 'messages': []}

    def sync_call():
        with httpx.Client(
            transport=PaidSyncTransport(
                runtime=runtime,
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
            )
        ) as client:
            return client.post('https://openrouter.ai/api/v1/chat/completions', json=body).json()

    try:
        assert (await asyncio.to_thread(sync_call)) == payload
        async with httpx.AsyncClient(
            transport=PaidAsyncTransport(
                runtime=runtime,
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
            )
        ) as client:
            await client.post('https://openrouter.ai/api/v1/chat/completions', json=body)
        calls = runtime.service.ledger.reconcile.await_args_list
        assert len(calls) == 2
        assert calls[0].args[2] == calls[1].args[2] == 20_000
        assert calls[0].kwargs == calls[1].kwargs
    finally:
        billing_principal.reset(token)


@pytest.mark.asyncio
async def test_stream_usage_saved_before_final_chunk(runtime):
    runtime.catalog.contract.return_value = ModelBillingContract(
        'tokens',
        'llm',
        100,
        frozenset({'model', 'max_tokens', 'messages', 'stream'}),
        'verified',
    )

    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"id":"gen-1","choices":[]}\n\n'
            yield b'data: {"id":"gen-1","usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n'
            yield b'data: [DONE]\n\n'

    token = billing_principal.set('user:1')
    try:
        async with httpx.AsyncClient(
            transport=PaidAsyncTransport(
                runtime=runtime,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200,
                        headers={'content-type': 'text/event-stream'},
                        stream=Chunks(),
                    )
                ),
            )
        ) as client:
            async with client.stream(
                'POST',
                'https://openrouter.ai/api/v1/chat/completions',
                json={
                    'model': 'm',
                    'max_tokens': 10,
                    'messages': [],
                    'stream': True,
                },
            ) as response:
                async for chunk in response.aiter_bytes():
                    if b'usage' in chunk:
                        runtime.service.ledger.reconcile.assert_awaited_once()
        assert runtime.service.ledger.reconcile.await_args.args[2] == 8000
    finally:
        billing_principal.reset(token)


@pytest.mark.asyncio
async def test_worker_restart_never_resends_pending_request(runtime):
    seen = set()
    dispatches = []

    async def reserve(**kwargs):
        operation_id = kwargs['operation_id']
        if operation_id in seen:
            return False
        seen.add(operation_id)
        return True

    async def provider(request):
        dispatches.append(True)
        raise httpx.ReadTimeout('unknown outcome')

    runtime.service.ledger.reserve.side_effect = reserve

    @bill_job_to_initiator
    async def run(ctx):
        async with httpx.AsyncClient(
            transport=PaidAsyncTransport(
                runtime=runtime,
                transport=httpx.MockTransport(provider),
            )
        ) as client:
            await client.post(
                'https://openrouter.ai/api/v1/chat/completions',
                json={
                    'model': 'm',
                    'max_tokens': 10,
                    'messages': [],
                },
            )

    with pytest.raises(httpx.ReadTimeout):
        await run({'job_id': 'job', 'job_try': 1}, cost_principal='user:1')
    runtime.rate.return_value = CostRate('openrouter', 'm', 2, 'tokens', Decimal('3'), Decimal('4'), 100)
    with pytest.raises(BudgetError, match='already reserved'):
        await run({'job_id': 'job', 'job_try': 2}, cost_principal='user:1')
    assert dispatches == [True]
    assert len(seen) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('sync', [False, True])
async def test_real_instructor_retry_accounts_invalid_response(runtime, monkeypatch, sync):
    runtime.loop = asyncio.get_running_loop()
    runtime.catalog = OpenRouterBillingCatalog()
    monkeypatch.setattr(runtime.catalog, 'openrouter_context', AsyncMock(return_value=100))
    calls = []

    def provider(request):
        body = json.loads(request.content)
        assert body['response_format']['type'] == 'json_object'
        calls.append(body)
        return httpx.Response(
            200,
            json={
                'id': f'gen-{len(calls)}',
                'object': 'chat.completion',
                'created': 1,
                'model': 'm',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {
                            'role': 'assistant',
                            'content': 'not JSON' if len(calls) == 1 else '{"value":1}',
                        },
                    }
                ],
                'usage': {'prompt_tokens': 4, 'completion_tokens': 2},
            },
        )

    monkeypatch.setattr(
        instructor_client,
        'PaidAsyncTransport',
        lambda **kwargs: PaidAsyncTransport(
            runtime=runtime,
            transport=httpx.MockTransport(provider),
        ),
    )
    monkeypatch.setattr(
        instructor_client,
        'PaidSyncTransport',
        lambda **kwargs: PaidSyncTransport(
            runtime=runtime,
            transport=httpx.MockTransport(provider),
        ),
    )

    class Result(BaseModel):
        value: int

    token = billing_principal.set('user:1')
    try:
        client = (
            instructor_client.create_instructor_client
            if sync
            else instructor_client.create_async_instructor_client
        )('https://openrouter.ai/api/v1', 'test')
        kwargs = {
            'model': 'm',
            'max_tokens': 10,
            'response_model': Result,
            'max_retries': 2,
            'messages': [{'role': 'user', 'content': 'Return value 1'}],
        }
        if sync:
            result = await asyncio.to_thread(client.chat.completions.create, **kwargs)
            await asyncio.to_thread(client.client.close)
        else:
            result = await client.chat.completions.create(**kwargs)
            await client.client.close()
        assert result.value == 1
        assert runtime.service.ledger.reserve.await_count == 2
        assert runtime.service.ledger.reconcile.await_count == 2
    finally:
        billing_principal.reset(token)


@pytest.mark.asyncio
async def test_cancellation_during_admission_refunds_without_dispatch(runtime):
    entered, release = asyncio.Event(), asyncio.Event()

    async def reserve(**kwargs):
        entered.set()
        await release.wait()
        return True

    runtime.service.ledger.reserve.side_effect = reserve
    provider = AsyncMock()
    token = billing_principal.set('user:1')
    try:
        async with httpx.AsyncClient(
            transport=PaidAsyncTransport(
                runtime=runtime,
                transport=httpx.MockTransport(provider),
            )
        ) as client:
            call = asyncio.create_task(
                client.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    json={
                        'model': 'm',
                        'max_tokens': 10,
                        'messages': [],
                    },
                )
            )
            await entered.wait()
            call.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await call
        provider.assert_not_called()
        assert runtime.service.ledger.reconcile.await_args.args[2] == 0
    finally:
        billing_principal.reset(token)


@pytest.mark.asyncio
async def test_received_usage_survives_cancellation(runtime):
    entered, release = asyncio.Event(), asyncio.Event()
    saved = []

    async def reconcile(*args, **kwargs):
        entered.set()
        await release.wait()
        saved.append(args[2])

    runtime.service.ledger.reconcile.side_effect = reconcile
    token = billing_principal.set('user:1')
    try:
        async with httpx.AsyncClient(
            transport=PaidAsyncTransport(
                runtime=runtime,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200,
                        json={
                            'id': 'gen-1',
                            'usage': {'cost': '0.0001'},
                        },
                    )
                ),
            )
        ) as client:
            call = asyncio.create_task(
                client.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    json={
                        'model': 'm',
                        'max_tokens': 10,
                        'messages': [],
                    },
                )
            )
            await entered.wait()
            call.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await call
        assert saved == [100_000]
        runtime.service.ledger.remember_response.assert_awaited_once()
    finally:
        billing_principal.reset(token)


@pytest.mark.asyncio
async def test_reconciliation_failure_is_reported_and_keeps_reservation(runtime):
    runtime.service.ledger.reconcile.side_effect = RuntimeError('database unavailable')
    token = billing_principal.set('user:1')
    try:
        async with httpx.AsyncClient(
            transport=PaidAsyncTransport(
                runtime=runtime,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200,
                        json={
                            'id': 'gen-1',
                            'usage': {'cost': '0.0001'},
                        },
                    )
                ),
            )
        ) as client:
            with pytest.raises(RuntimeError, match='database unavailable'):
                await client.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    json={
                        'model': 'm',
                        'max_tokens': 10,
                        'messages': [],
                    },
                )
        runtime.service.ledger.reserve.assert_awaited_once()
        runtime.service.ledger.reconcile.assert_awaited_once()
    finally:
        billing_principal.reset(token)


@pytest.mark.asyncio
async def test_managed_llm_uses_one_physical_reservation(runtime, monkeypatch):
    runtime.catalog = OpenRouterBillingCatalog()
    monkeypatch.setattr(runtime.catalog, 'openrouter_context', AsyncMock(return_value=100))

    def provider(request):
        return httpx.Response(
            200,
            json={
                'id': 'gen-managed',
                'object': 'chat.completion',
                'created': 1,
                'model': 'm',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {
                            'role': 'assistant',
                            'content': 'OK',
                        },
                    }
                ],
                'usage': {'prompt_tokens': 4, 'completion_tokens': 2, 'cost': '0.00001'},
            },
        )

    monkeypatch.setattr(
        factories,
        'PaidAsyncTransport',
        lambda **kwargs: PaidAsyncTransport(
            runtime=runtime,
            transport=httpx.MockTransport(provider),
        ),
    )
    llm = ManagedLLM(
        factories.create_openrouter_llm(
            runtime=runtime,
            max_tokens=10,
            model=factories.settings.model_copy(
                update={'openrouter_model': 'm', 'openrouter_api_key': 'test'},
            ),
        )
    )
    token = billing_principal.set('user:1')
    try:
        response = await llm.ainvoke('hello')
        assert response.content == 'OK'
        runtime.service.ledger.reserve.assert_awaited_once()
        runtime.service.ledger.reconcile.assert_awaited_once()
    finally:
        billing_principal.reset(token)
        await llm.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('grouped', [False, True])
async def test_both_judge_factories_use_physical_accounting(runtime, monkeypatch, grouped):
    runtime.loop = asyncio.get_running_loop()
    runtime.catalog = OpenRouterBillingCatalog()
    monkeypatch.setattr(runtime.catalog, 'openrouter_context', AsyncMock(return_value=100))
    monkeypatch.setattr(judge.settings, 'llm_provider', LLMProvider.OPENROUTER)
    monkeypatch.setattr(judge.settings, 'openrouter_api_key', 'test')
    monkeypatch.setattr(judge.settings, 'benchmark_judge_initial_tokens', 10)
    content = (
        '{"relevancy":{"score":8,"reason":"grounded"}}' if grouped else '{"score":8,"reason":"grounded"}'
    )

    def provider(request):
        return httpx.Response(
            200,
            json={
                'id': 'gen-judge',
                'object': 'chat.completion',
                'created': 1,
                'model': 'm',
                'choices': [
                    {
                        'index': 0,
                        'finish_reason': 'stop',
                        'message': {
                            'role': 'assistant',
                            'content': content,
                        },
                    }
                ],
                'usage': {'prompt_tokens': 4, 'completion_tokens': 2},
            },
        )

    monkeypatch.setattr(
        judge_clients,
        'PaidSyncTransport',
        lambda **kwargs: PaidSyncTransport(
            runtime=runtime,
            transport=httpx.MockTransport(provider),
        ),
    )
    monkeypatch.setattr(
        judge_clients,
        'PaidAsyncTransport',
        lambda **kwargs: PaidAsyncTransport(
            runtime=runtime,
            transport=httpx.MockTransport(provider),
        ),
    )
    token = billing_principal.set('user:1')
    try:
        if grouped:
            saved = {}

            async def checkpoint(value):
                pass

            client = judge_clients.create_async_judge_client()
            try:
                await judge_requests.judge_batch(
                    client,
                    'm',
                    {'question': 'q', 'answer': 'a'},
                    ['relevancy'],
                    saved,
                    checkpoint,
                    budget=10,
                    attempt=1,
                )
                assert saved['relevancy']['score'] == 8
            finally:
                await client.close()
        else:
            judge_clients.retire_judge_clients()
            client = await asyncio.to_thread(judge.get_judge_client, 'm')
            result = await asyncio.to_thread(judge.judge_with_structured_output, client, 'score this', 'm')
            assert result.score == 8
            await judge_clients.close_judge_clients()
        runtime.service.ledger.reserve.assert_awaited_once()
        runtime.service.ledger.reconcile.assert_awaited_once()
    finally:
        billing_principal.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize('sync,rerank', [(False, False), (True, False), (False, True), (True, True)])
async def test_deepinfra_clients_keep_outputs_and_account_physical_call(runtime, monkeypatch, sync, rerank):
    runtime.loop = asyncio.get_running_loop()
    operation = 'reranking' if rerank else 'embedding'
    runtime.catalog.contract.return_value = ModelBillingContract(
        'tokens',
        operation,
        100,
        frozenset({'queries', 'documents'} if rerank else {'model', 'input', 'encoding_format'}),
        'verified',
    )
    runtime.rate.return_value = CostRate(
        'deepinfra', 'm', 1, 'tokens', Decimal('1'), Decimal('0'), 100, input_only=True
    )
    payload = (
        {'scores': [0.9], 'input_tokens': 7}
        if rerank
        else {
            'data': [{'index': 0, 'embedding': [0.1, 0.2]}],
            'usage': {'prompt_tokens': 7},
        }
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    monkeypatch.setattr(
        managed_http,
        'PaidAsyncTransport',
        lambda **kwargs: PaidAsyncTransport(
            runtime=runtime,
            transport=transport,
        ),
    )
    monkeypatch.setattr(
        deepinfra_clients,
        'PaidSyncTransport',
        lambda **kwargs: PaidSyncTransport(
            runtime=runtime,
            transport=transport,
        ),
    )
    client = (
        deepinfra_clients.DeepInfraRerankerClient if rerank else deepinfra_clients.DeepInfraEmbeddingsClient
    )(
        'test',
        'https://api.deepinfra.com/v1' + ('' if rerank else '/openai'),
        'm',
    )
    token = billing_principal.set('user:1')
    try:
        if rerank:
            result = (
                await asyncio.to_thread(client.predict_sync, [('q', 'd')])
                if sync
                else (await client.predict([('q', 'd')]))
            )
            assert result == [0.9]
        else:
            result = (
                await asyncio.to_thread(client.embed_query_sync, 'q')
                if sync
                else (await client.embed_query('q'))
            )
            assert result == [0.1, 0.2]
        runtime.service.ledger.reserve.assert_awaited_once()
        assert runtime.service.ledger.reconcile.await_args.args[2] == 7000
    finally:
        billing_principal.reset(token)
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['unknown_price', 'missing_usage', 'invalid_usage'])
async def test_enabled_quota_does_not_make_unknown_billing_free(runtime, failure):
    if failure == 'unknown_price':
        runtime.rate.side_effect = BudgetError('Unknown tariff')
    payload = {'usage': {'cost': 'NaN'}} if failure == 'invalid_usage' else {}
    provider = AsyncMock(return_value=httpx.Response(200, json=payload))
    token = billing_principal.set('user:1')
    try:
        async with httpx.AsyncClient(
            transport=PaidAsyncTransport(
                runtime=runtime,
                transport=httpx.MockTransport(provider),
            )
        ) as client:
            if failure == 'unknown_price':
                with pytest.raises(BudgetError):
                    await client.post(
                        'https://openrouter.ai/api/v1/chat/completions',
                        json={
                            'model': 'm',
                            'max_tokens': 10,
                            'messages': [],
                        },
                    )
                provider.assert_not_called()
                runtime.service.ledger.reserve.assert_not_called()
            else:
                await client.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    json={
                        'model': 'm',
                        'max_tokens': 10,
                        'messages': [],
                    },
                )
                runtime.service.ledger.reserve.assert_awaited_once()
                runtime.service.ledger.reconcile.assert_not_called()
    finally:
        billing_principal.reset(token)


@pytest.mark.asyncio
async def test_instructor_retry_cannot_bypass_worker_replay_block(runtime, monkeypatch):
    runtime.catalog = OpenRouterBillingCatalog()
    monkeypatch.setattr(runtime.catalog, 'openrouter_context', AsyncMock(return_value=100))
    seen = set()
    dispatches = []

    async def reserve(**kwargs):
        operation_id = kwargs['operation_id']
        if operation_id in seen:
            return False
        seen.add(operation_id)
        return True

    def provider(request):
        dispatches.append(True)
        raise httpx.ReadTimeout('ambiguous outcome')

    runtime.service.ledger.reserve.side_effect = reserve
    monkeypatch.setattr(
        instructor_client,
        'PaidAsyncTransport',
        lambda **kwargs: PaidAsyncTransport(
            runtime=runtime,
            transport=httpx.MockTransport(provider),
        ),
    )
    client = instructor_client.create_async_instructor_client('https://openrouter.ai/api/v1', 'test')

    class Result(BaseModel):
        value: int

    @bill_job_to_initiator
    async def run(ctx, retries):
        return await client.chat.completions.create(
            model='m',
            max_tokens=10,
            max_retries=retries,
            response_model=Result,
            messages=[{'role': 'user', 'content': 'Return value 1'}],
        )

    try:
        with pytest.raises(InstructorRetryException):
            await run({'job_id': 'restarted-job'}, 1, cost_principal='user:1')
        assert dispatches == [True]
        with pytest.raises(InstructorRetryException):
            await run({'job_id': 'restarted-job'}, 2, cost_principal='user:1')
        assert dispatches == [True]
    finally:
        await client.client.close()


@pytest.mark.asyncio
async def test_worker_operation_ids_follow_requests_when_parallel_order_changes(runtime):
    observed = {}
    token = billing_principal.set('user:1')

    @bill_job_to_initiator
    async def run(ctx, questions):
        for question in questions:
            async with httpx.AsyncClient(transport=PaidAsyncTransport(
                runtime=runtime,
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
            )) as client:
                await client.post('https://openrouter.ai/api/v1/chat/completions', json={
                    'model': 'm', 'max_tokens': 10,
                    'messages': [{'role': 'user', 'content': question}],
                })
            observed[(ctx['job_try'], question)] = (
                runtime.service.ledger.reserve.await_args.kwargs['operation_id']
            )

    try:
        await run({'job_id': 'job', 'job_try': 1}, ['a', 'b'], cost_principal='user:1')
        await run({'job_id': 'job', 'job_try': 2}, ['b', 'a'], cost_principal='user:1')
        assert observed[(1, 'a')] == observed[(2, 'a')]
        assert observed[(1, 'b')] == observed[(2, 'b')]
        assert observed[(1, 'a')] != observed[(1, 'b')]
    finally:
        billing_principal.reset(token)


@pytest.mark.asyncio
async def test_cancelled_stream_retains_response_id_without_refund(runtime):
    runtime.catalog.contract.return_value = ModelBillingContract(
        'tokens', 'llm', 100,
        frozenset({'model', 'max_tokens', 'messages', 'stream'}), 'verified',
    )

    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"id":"gen-cancelled","choices":[]}\n\n'
            raise asyncio.CancelledError

    token = billing_principal.set('user:1')
    try:
        async with httpx.AsyncClient(transport=PaidAsyncTransport(
            runtime=runtime, transport=httpx.MockTransport(lambda request: httpx.Response(
                200, headers={'content-type': 'text/event-stream'}, stream=Chunks(),
            )),
        )) as client:
            with pytest.raises(asyncio.CancelledError):
                await client.post('https://openrouter.ai/api/v1/chat/completions', json={
                    'model': 'm', 'max_tokens': 10, 'messages': [], 'stream': True,
                })
        runtime.service.ledger.reserve.assert_awaited_once()
        runtime.service.ledger.reconcile.assert_not_called()
        assert runtime.service.ledger.remember_response.await_args.args[2] == 'gen-cancelled'
    finally:
        billing_principal.reset(token)
