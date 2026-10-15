"""Provider admission shared by chat, auxiliary calls and benchmark adapters."""

import json
import hashlib
from contextvars import ContextVar
from contextlib import asynccontextmanager

from sqlalchemy import text
import psycopg
from langchain_core.load.dump import dumpd
from domain.utils import parse_bool

from application.ports.billing_context import billing_principal
from infrastructure.budget.job_context import next_operation_id
from application.services.cost_budget_service import BudgetPolicy, CostBudgetService
from config import settings
from domain.value_objects.cost_budget import BudgetError, CostRate
from infrastructure.ml.usage_capture import active_llm_usage
from infrastructure.storage.file_storage import finish_owned_task
import asyncio


paid_attempt: ContextVar[dict | None] = ContextVar('paid_attempt', default=None)


class BudgetRuntime:
    def __init__(self):
        self.service = None
        self.session_factory = None
        self.catalog = None
        self.bootstrapped = False
        self.loop = None

    def configure(self, ledger, session_factory, catalog):
        self.loop = asyncio.get_running_loop()
        self.service = CostBudgetService(ledger)
        self.session_factory = session_factory
        self.catalog = catalog

    def run_sync(self, function, *args):
        """Use the Composition-owned async resources from a worker thread."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise BudgetError('Synchronous paid clients must run outside the event loop')
        if self.loop is None or not self.loop.is_running():
            raise BudgetError('Cost backend event loop is unavailable')
        return asyncio.run_coroutine_threadsafe(function(*args), self.loop).result()

    def decode_policy(self, rows):
        values = dict(rows)
        if not {'cost_rate_limit_enabled', 'cost_hourly_limit', 'cost_daily_limit'} <= values.keys():
            raise BudgetError('Budget configuration is not initialized')
        return BudgetPolicy(
            parse_bool(values['cost_rate_limit_enabled']),
            values['cost_hourly_limit'],
            values['cost_daily_limit'],
        )

    async def policy_snapshot(self):
        if not self.bootstrapped:
            return BudgetPolicy(
                settings.cost_rate_limit_enabled,
                str(settings.cost_hourly_limit),
                str(settings.cost_daily_limit),
            )
        if self.session_factory is None:
            raise BudgetError('Cost backend is not initialized')
        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    text('''SELECT key, value FROM config_parameters
                WHERE domain_key IS NULL AND key IN
                ('cost_rate_limit_enabled','cost_hourly_limit','cost_daily_limit')''')
                )
            ).all()
        return self.decode_policy(rows)

    def policy_sync(self):
        if not self.bootstrapped:
            return BudgetPolicy(
                settings.cost_rate_limit_enabled,
                str(settings.cost_hourly_limit),
                str(settings.cost_daily_limit),
            )
        with psycopg.connect(
            host=settings.db_host,
            port=int(settings.db_port),
            user=settings.db_user,
            password=settings.db_password,
            dbname=settings.db_name,
            connect_timeout=5,
            options='-c statement_timeout=5000',
            autocommit=True,
        ) as connection:
            rows = connection.execute('''SELECT key, value FROM config_parameters
                WHERE domain_key IS NULL AND key IN
                ('cost_rate_limit_enabled','cost_hourly_limit','cost_daily_limit')''').fetchall()
        return self.decode_policy(rows)

    def reject_unsupported(self, operation):
        raise BudgetError(f'USD accounting unsupported for {operation}; paid call blocked')

    async def reject_unsupported_async(self, operation):
        raise BudgetError(f'USD accounting unsupported for {operation}; paid call blocked')

    async def rate(self, provider, model):
        if self.session_factory is None:
            raise BudgetError('Cost backend is not initialized')
        async with self.session_factory() as session:
            row = (
                (
                    await session.execute(
                        text('''
                SELECT provider, model, version, unit, input_usd_per_million,
                       output_usd_per_million, context_tokens FROM cost_rates
                WHERE provider=:provider AND model=:model ORDER BY version DESC LIMIT 1
            '''),
                        {'provider': provider, 'model': model},
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise BudgetError('Unknown provider/model price; paid call blocked')
        rate = CostRate(**row)
        if self.catalog is None or rate.context_tokens < await self.catalog.max_context(provider, model):
            raise BudgetError('Provider context bound changed; publish a verified tariff version')
        async with self.session_factory() as session:
            violated = (
                await session.execute(
                    text('''SELECT EXISTS(SELECT 1 FROM cost_usage
                WHERE provider=:provider AND model=:model AND rate_version=:version
                AND actual_units > reserved_units)'''),
                    {'provider': provider, 'model': model, 'version': rate.version},
                )
            ).scalar_one()
        if violated:
            raise BudgetError('This tariff exceeded its reservation bound; paid call blocked')
        return rate

    def bounded_cost(self, client, kwargs, model, rate):
        output = kwargs.get('max_tokens') or getattr(client, 'max_tokens', None)
        if not isinstance(output, int) or isinstance(output, bool):
            raise BudgetError('Finite output budget required')
        if model == 'openrouter/auto' or model.endswith(':online'):
            raise BudgetError('Automatic routing/web billing is unsupported')
        maximum = rate.reserve(rate.context_tokens, output)
        if kwargs.get('tools') or kwargs.get('functions') or kwargs.get('extra_body'):
            raise BudgetError('Unsupported paid request extensions')
        return output, maximum

    @asynccontextmanager
    async def llm_call(self, client, kwargs, *, input=None):
        # Detect the actual leased client, never the current global provider setting.
        if not hasattr(client, 'openai_api_base'):
            yield kwargs
            return
        policy = await self.policy_snapshot()
        if not policy.enabled:
            yield kwargs
            return
        if self.service is None:
            raise BudgetError('Cost backend is not initialized')
        model = kwargs.get('model') or getattr(client, 'model_name', '')
        rate = await self.rate('openrouter', model)
        output, maximum = self.bounded_cost(client, kwargs, model, rate)
        principal = billing_principal.get()
        request_key = hashlib.sha256(
            json.dumps(dumpd({'model': model, 'input': input, 'kwargs': kwargs}), sort_keys=True).encode()
        ).hexdigest()
        operation_id = next_operation_id(request_key)
        records: list[dict] = []
        previous = active_llm_usage.get()
        token = active_llm_usage.set(records)
        attempt = {"model": model, "output": output, "sent": False}
        attempt_token = paid_attempt.set(attempt)
        guarded = dict(
            kwargs,
            extra_body={
                'provider': {
                    'require_parameters': True,
                    'max_price': {
                        'prompt': str(rate.input_usd_per_million),
                        'completion': str(rate.output_usd_per_million),
                        'request': 0,
                        'image': 0,
                    },
                },
            },
        )
        attempt["provider"] = guarded["extra_body"]["provider"]
        reserved = False
        try:
            async with self.service.operation(
                policy=policy,
                principal=principal,
                operation_id=operation_id,
                provider='openrouter',
                model=model,
                operation='llm',
                maximum_units=maximum,
                rate_version=rate.version,
            ):
                reserved = True
                yield guarded
                if len(records) == 1 and records[0].get('cost_usd') is not None:
                    await self.service.settle(principal, operation_id, str(records[0]['cost_usd']))
        finally:
            try:
                if reserved and not attempt['sent']:
                    await finish_owned_task(
                        asyncio.create_task(self.service.settle(principal, operation_id, '0'))
                    )
                if reserved and records and records[-1].get('response_id'):
                    await finish_owned_task(
                        asyncio.create_task(
                            self.service.ledger.remember_response(
                                principal, operation_id, records[-1]['response_id']
                            )
                        )
                    )
            finally:
                paid_attempt.reset(attempt_token)
                active_llm_usage.reset(token)
                if previous is not None:
                    previous.extend(records)


budget_runtime = BudgetRuntime()


def guard_paid_http(request):
    attempt = paid_attempt.get()
    validate_paid_request(request, attempt)


def validate_paid_request(request, attempt):
    if attempt is None or attempt['sent']:
        raise BudgetError('Paid request has no unused durable reservation')
    body = json.loads(request.content)
    if (
        request.url.host != 'openrouter.ai'
        or request.url.scheme != 'https'
        or body.get('provider') != attempt['provider']
        or body.get('model') != attempt['model']
        or body.get('max_tokens', body.get('max_completion_tokens')) != attempt['output']
        or body.get('n', 1) != 1
        or any(body.get(key) for key in ('tools', 'functions', 'plugins', 'models', 'audio', 'modalities'))
        or any(not isinstance(message.get('content'), str) for message in body.get('messages', []))
    ):
        raise BudgetError('Unsupported paid request shape')
    attempt['sent'] = True


async def guard_paid_http_async(request):
    attempt = paid_attempt.get()
    validate_paid_request(request, attempt)
