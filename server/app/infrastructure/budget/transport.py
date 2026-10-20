"""One durable reservation per physical HTTP attempt, shared by sync/async SDKs."""

import asyncio
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import logging
from typing import TypeGuard

import httpx

from application.budget import billing_principal
from domain.runtime.cost import BudgetError, CostRate, UNDISPATCHED_COST_SOURCE, usd_units
from infrastructure.budget.job_context import (
    next_operation_id,
    block_job_dispatch,
    ensure_job_dispatch_allowed,
)
from infrastructure.budget.runtime import BudgetRuntime, budget_runtime
from infrastructure.storage.file_storage import finish_owned_task

log = logging.getLogger('default')


async def persist_accounting(coroutine):
    task = asyncio.create_task(coroutine)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await finish_owned_task(task)
        raise


def valid_token_count(value) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass
class PhysicalAttempt:
    runtime: BudgetRuntime
    principal: str
    operation_id: str
    rate: CostRate
    operation: str
    response_id: str | None = None

    async def release_before_dispatch(self):
        if self.runtime.service is None:
            raise BudgetError('Cost backend is not initialized')
        await persist_accounting(
            self.runtime.service.ledger.reconcile(
                self.principal, self.operation_id, 0, cost_source=UNDISPATCHED_COST_SOURCE
            )
        )

    async def save(self, payload):
        if self.runtime.service is None:
            raise BudgetError("Cost backend is not initialized")
        ledger = self.runtime.service.ledger
        response_id = payload.get('id') or payload.get('request_id')
        if isinstance(response_id, str) and response_id and response_id != self.response_id:
            await ledger.remember_response(self.principal, self.operation_id, response_id)
            self.response_id = response_id
        usage = payload.get('usage') or {}
        status = payload.get('inference_status') or {}
        if not isinstance(usage, dict) or not isinstance(status, dict):
            return
        inputs = payload.get('input_tokens') if self.operation == 'reranking' else usage.get('prompt_tokens')
        outputs = usage.get('completion_tokens') if self.operation == 'llm' else 0
        units = {'input_tokens': inputs, 'output_tokens': outputs}
        cost = usage.get('cost', status.get('cost'))
        source = 'provider'
        if cost is not None:
            try:
                actual = usd_units(str(cost))
            except BudgetError:
                return
        elif valid_token_count(inputs) and valid_token_count(outputs):
            # Actual usage may exceed the bound: record the entire debt before blocking the tariff.
            actual = usd_units(
                (
                    int(inputs) * self.rate.input_usd_per_million
                    + int(outputs) * self.rate.output_usd_per_million
                )
                / 1_000_000
            )
            source = 'tariff'
        else:
            return
        await ledger.reconcile(
            self.principal, self.operation_id, actual, billable_units=units, cost_source=source
        )

    async def persist(self, payload):
        try:
            await persist_accounting(self.save(payload))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('Paid usage persistence failed; inspect reservation %s', self.operation_id)
            raise


async def admit_request(runtime, request, policy):
    if runtime.service is None or runtime.catalog is None:
        raise BudgetError('Cost backend is not initialized')
    if request.method != 'POST' or request.url.scheme != 'https':
        raise BudgetError('Unsupported paid HTTP endpoint')
    body = json.loads(request.content)
    if not isinstance(body, dict):
        raise BudgetError('Unsupported paid request')
    provider, model = paid_endpoint(request, body)
    contract = await runtime.catalog.contract(provider, model)
    if set(body) - contract.allowed_parameters:
        raise BudgetError('Unsupported paid request extensions')
    rate = await runtime.rate(provider, model)
    if contract.unit != rate.unit or rate.context_tokens < contract.input_limit:
        raise BudgetError('Tariff does not cover the verified model contract')
    request_body = dict(body)
    maximum = prepare_body(body, contract, rate)
    fingerprint = hashlib.sha256(
        json.dumps(
            {'provider': provider, 'model': model, 'operation': contract.operation, 'body': request_body},
            sort_keys=True,
            separators=(',', ':'),
        ).encode()
    ).hexdigest()
    operation_id = next_operation_id(fingerprint)
    principal = billing_principal.get()
    if not principal or not principal.startswith(('user:', 'system:')):
        raise BudgetError('Paid operation requires an explicit billing principal')
    hour, day = usd_units(policy.hourly_usd), usd_units(policy.daily_usd)
    if policy.enabled and (not hour or not day):
        raise BudgetError('Enabled cost limits must be positive')
    admitted = await runtime.service.ledger.reserve(
        principal=principal,
        operation_id=operation_id,
        provider=provider,
        model=model,
        operation=contract.operation,
        maximum_units=maximum,
        rate_version=rate.version,
        hourly_units=hour,
        daily_units=day,
        enforce_limits=policy.enabled,
        request_fingerprint=fingerprint,
    )
    if not admitted:
        block_job_dispatch()
        raise BudgetError('Operation already reserved; reconcile before retrying')
    attempt = PhysicalAttempt(runtime, principal, operation_id, rate, contract.operation)
    try:
        content = json.dumps(body).encode()
        headers = request.headers.copy()
        headers['Content-Length'] = str(len(content))
        prepared = httpx.Request(
            request.method, request.url, headers=headers, content=content, extensions=request.extensions
        )
    except BaseException:
        await attempt.release_before_dispatch()
        raise
    return attempt, prepared


def paid_endpoint(request, body):
    if request.url.host == 'openrouter.ai' and request.url.path == '/api/v1/chat/completions':
        provider, model = 'openrouter', body.get('model')
    elif request.url.host == 'api.deepinfra.com' and request.url.path == '/v1/openai/embeddings':
        provider, model = 'deepinfra', body.get('model')
    elif request.url.host == 'api.deepinfra.com' and request.url.path.startswith('/v1/inference/'):
        provider, model = 'deepinfra', request.url.path.removeprefix('/v1/inference/')
    else:
        raise BudgetError('Unsupported paid HTTP endpoint')
    return provider, model


def prepare_body(body, contract, rate):
    if contract.operation == 'llm':
        return prepare_llm_body(body, contract, rate)
    return prepare_inference_body(body, contract, rate)


def prepare_llm_body(body, contract, rate):
    output = body.get('max_tokens', body.get('max_completion_tokens'))
    if (
        not isinstance(output, int)
        or isinstance(output, bool)
        or output <= 0
        or body.get('n', 1) != 1
        or body.get('provider')
        or not isinstance(body.get('messages'), list)
        or any(not isinstance(m, dict) or not isinstance(m.get('content'), str) for m in body['messages'])
        or body.get('response_format', {}).get('type', 'text') not in ('text', 'json_object')
    ):
        raise BudgetError('Unsupported paid request shape or missing output bound')
    maximum = rate.reserve(contract.input_limit, output)
    body['provider'] = {
        'require_parameters': True,
        'max_price': {
            'prompt': str(rate.input_usd_per_million),
            'completion': str(rate.output_usd_per_million),
            'request': 0,
            'image': 0,
        },
    }
    if body.get('stream'):
        body['stream_options'] = {'include_usage': True}
    return maximum


def prepare_inference_body(body, contract, rate):
    if contract.operation == 'embedding':
        elements = body.get('input')
        elements = [elements] if isinstance(elements, str) else elements
        if body.get('encoding_format', 'float') != 'float':
            raise BudgetError('Unsupported embedding encoding')
    else:
        queries = body.get('queries')
        if not isinstance(queries, list) or len(queries) != 1 or not isinstance(queries[0], str):
            raise BudgetError('One reranking query per physical request is required')
        elements = body.get('documents')
    if not isinstance(elements, list) or not elements or any(not isinstance(v, str) for v in elements):
        raise BudgetError('Invalid inference batch')
    # Each element/pair reserves the entire model bound, never a text-length estimate.
    maximum = len(elements) * rate.reserve(contract.input_limit, 0)
    return maximum


class PaidResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream, attempt, streaming):
        self.stream, self.attempt, self.streaming = stream, attempt, streaming
        self.buffer = b''

    async def __aiter__(self):
        async for chunk in self.stream:
            self.buffer += chunk
            if self.streaming:
                while b'\n' in self.buffer:
                    line, self.buffer = self.buffer.split(b'\n', 1)
                    if line.startswith(b'data:'):
                        await self.consume(line[5:].strip())
            yield chunk
        if not self.streaming:
            await self.consume(self.buffer)
        elif self.buffer.startswith(b'data:'):
            await self.consume(self.buffer[5:].strip())

    async def consume(self, data):
        try:
            payload = json.loads(data, parse_float=Decimal)
        except (ValueError, UnicodeDecodeError):
            return  # Missing/malformed confirmation retains the reservation.
        if isinstance(payload, dict):
            await self.attempt.persist(payload)

    async def aclose(self):
        await self.stream.aclose()


class PaidAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, runtime=budget_runtime, transport=None):
        self.runtime = runtime
        self.transport = transport or httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request):
        policy = await self.runtime.policy_snapshot()
        if not policy.enabled:
            return await self.transport.handle_async_request(request)
        admission = asyncio.create_task(admit_request(self.runtime, request, policy))
        try:
            attempt, request = await asyncio.shield(admission)
        except asyncio.CancelledError:
            try:
                attempt, _ = await finish_owned_task(admission)
            except Exception:
                log.exception('Cancelled admission failed before HTTP dispatch')
            else:
                await attempt.release_before_dispatch()
            raise
        try:
            ensure_job_dispatch_allowed()
            await persist_accounting(
                self.runtime.service.ledger.mark_dispatch(
                    attempt.principal,
                    attempt.operation_id,
                )
            )
            ensure_job_dispatch_allowed()
        except BaseException:
            # Transport has not been called, so this attempt is provably unbilled.
            await attempt.release_before_dispatch()
            raise
        response = await self.transport.handle_async_request(request)
        if 'text/event-stream' in response.headers.get('content-type', ''):
            response.stream = PaidResponseStream(response.stream, attempt, True)
        else:
            try:
                data = await response.aread()
                consumer = PaidResponseStream(response.stream, attempt, False)
                await consumer.consume(data)
            finally:
                await response.aclose()
        return response

    async def aclose(self):
        await self.transport.aclose()


class PaidSyncTransport(httpx.BaseTransport):
    """Sync I/O remains on its caller thread; only ledger work uses the owner loop."""

    def __init__(self, *, runtime=budget_runtime, transport=None):
        self.runtime = runtime
        self.transport = transport or httpx.HTTPTransport(retries=0)

    def handle_request(self, request):
        policy = (
            self.runtime.policy_sync()
            if self.runtime.loop is None
            else self.runtime.run_sync(self.runtime.policy_snapshot)
        )
        if not policy.enabled:
            return self.transport.handle_request(request)
        attempt, prepared = self.runtime.run_sync(admit_request, self.runtime, request, policy)
        try:
            ensure_job_dispatch_allowed()
            self.runtime.run_sync(
                self.runtime.service.ledger.mark_dispatch, attempt.principal, attempt.operation_id
            )
            ensure_job_dispatch_allowed()
        except BaseException:
            self.runtime.run_sync(attempt.release_before_dispatch)
            raise
        response = self.transport.handle_request(prepared)
        try:
            payload = json.loads(response.read(), parse_float=Decimal)
            if isinstance(payload, dict):
                self.runtime.run_sync(attempt.persist, payload)
        except (ValueError, UnicodeDecodeError):
            pass  # Unknown billing response remains pending.
        finally:
            response.close()
        return response

    def close(self):
        self.transport.close()
