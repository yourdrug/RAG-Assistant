"""PostgreSQL is durable truth; Redis admission is rebuilt under a principal lock.

No provider I/O occurs inside these short transactions. Every reservation rebuilds
Redis from committed ledger rows, so a Redis restart cannot grant free credit.
Pending outcomes count in both windows until explicitly reconciled.
"""

import hashlib
import json

from sqlalchemy import text

from domain.value_objects.cost_budget import BudgetError, UNDISPATCHED_COST_SOURCE

ADMIT = """
local hour = tonumber(ARGV[1]) + tonumber(ARGV[3])
local day = tonumber(ARGV[2]) + tonumber(ARGV[3])
if hour > tonumber(ARGV[4]) or day > tonumber(ARGV[5]) then return 0 end
redis.call('HSET', KEYS[1], 'hour', hour, 'day', day, 'operation', ARGV[6])
redis.call('EXPIRE', KEYS[1], 86400)
return 1
"""


class PostgresRedisBudget:
    def __init__(self, session_factory, redis):
        self.session_factory = session_factory
        self.redis = redis

    async def lock(self, session, principal, *, shared=False):
        digest = hashlib.sha256(principal.encode()).digest()
        lock_id = int.from_bytes(digest[:8], 'big', signed=True)
        statement = (
            'SELECT pg_advisory_xact_lock_shared(:key)' if shared else 'SELECT pg_advisory_xact_lock(:key)'
        )
        await session.execute(text(statement), {'key': lock_id})

    async def reserve(
        self,
        *,
        principal,
        operation_id,
        provider,
        model,
        operation,
        maximum_units,
        rate_version,
        hourly_units,
        daily_units,
        enforce_limits=True,
        request_fingerprint=None,
    ):
        async with self.session_factory() as session, session.begin():
            await self.lock(session, f'tariff:{provider}:{model}:{rate_version}', shared=True)
            await self.lock(session, principal)
            violated = (
                await session.execute(
                    text("""SELECT EXISTS(SELECT 1 FROM cost_usage
                    WHERE provider=:provider AND model=:model AND rate_version=:version
                    AND actual_units > reserved_units)"""),
                    {'provider': provider, 'model': model, 'version': rate_version},
                )
            ).scalar_one()
            if violated:
                raise BudgetError('Violated tariff; publish a verified new version')
            existing = (
                (
                    await session.execute(
                        text('''SELECT principal, provider, model, operation, cost_source,
                            actual_units, dispatch_state, request_fingerprint, provider_response_id
                            FROM cost_usage WHERE operation_id=:id'''),
                        {'id': operation_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if existing is not None:
                if existing['principal'] != principal:
                    raise BudgetError('Operation ID belongs to another principal')
                # Only a proven absence of HTTP dispatch permits rearming this request.
                if not (
                    existing['cost_source'] == UNDISPATCHED_COST_SOURCE
                    and existing['actual_units'] == 0
                    and existing['dispatch_state'] == 'settled'
                    and existing['provider_response_id'] is None
                    and request_fingerprint is not None
                    and existing['request_fingerprint'] == request_fingerprint
                    and (existing['provider'], existing['model'], existing['operation'])
                    == (provider, model, operation)
                ):
                    return False
            totals = (
                (
                    await session.execute(
                        text('''
                SELECT COALESCE(SUM(COALESCE(actual_units, reserved_units)) FILTER (
                    WHERE settled_at IS NULL OR settled_at > now() - interval '1 hour'), 0) AS hour,
                    COALESCE(SUM(COALESCE(actual_units, reserved_units)), 0) AS day
                FROM cost_usage WHERE principal=:principal
                AND (settled_at IS NULL OR settled_at > now() - interval '1 day')
            '''),
                        {'principal': principal},
                    )
                )
                .mappings()
                .one()
            )
            key = 'rag:budget:v1:' + hashlib.sha256(principal.encode()).hexdigest()
            admitted = not enforce_limits or await self.redis.eval(
                ADMIT,
                1,
                key,
                int(totals['hour']),
                int(totals['day']),
                maximum_units,
                hourly_units,
                daily_units,
                operation_id,
            )
            if not admitted:
                raise BudgetError('Rolling USD budget exceeded')
            statement = '''
                INSERT INTO cost_usage (operation_id, principal, provider, model, operation,
                                        rate_version, reserved_units, request_fingerprint)
                VALUES (:id, :principal, :provider, :model, :operation, :version, :units, :fingerprint)
            '''
            if existing is not None:
                statement = '''UPDATE cost_usage SET rate_version=:version, reserved_units=:units,
                    actual_units=NULL, settled_at=NULL, billable_units=NULL, cost_source=NULL,
                    reconciliation_evidence=NULL, dispatch_state='reserved', created_at=now()
                    WHERE operation_id=:id'''
            await session.execute(
                text(statement),
                {
                    "id": operation_id,
                    "principal": principal,
                    "provider": provider,
                    "model": model,
                    "operation": operation,
                    "version": rate_version,
                    "units": maximum_units,
                    "fingerprint": request_fingerprint,
                },
            )
        return True

    async def reconcile(
        self, principal, operation_id, actual_units, evidence=None, *, billable_units=None, cost_source=None
    ):
        if isinstance(actual_units, bool) or not isinstance(actual_units, int) or actual_units < 0:
            raise BudgetError("Invalid actual charge")
        async with self.session_factory() as session, session.begin():
            tariff = (
                await session.execute(
                    text("""SELECT provider, model, rate_version FROM cost_usage
                    WHERE operation_id=:id AND principal=:principal"""),
                    {'id': operation_id, 'principal': principal},
                )
            ).one_or_none()
            if tariff is None:
                raise BudgetError('Unknown reservation')
            await self.lock(session, f'tariff:{tariff.provider}:{tariff.model}:{tariff.rate_version}')
            await self.lock(session, principal)
            row = (
                (
                    await session.execute(
                        text('''
                SELECT reserved_units, actual_units, billable_units, cost_source FROM cost_usage
                WHERE operation_id=:id AND principal=:principal FOR UPDATE
            '''),
                        {'id': operation_id, 'principal': principal},
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise BudgetError('Unknown reservation')
            if row['actual_units'] is not None:
                if (
                    row['actual_units'] != actual_units
                    or (billable_units is not None and row['billable_units'] != billable_units)
                    or (cost_source is not None and row['cost_source'] != cost_source)
                ):
                    raise BudgetError('Conflicting usage reconciliation')
                return
            await session.execute(
                text('''
                UPDATE cost_usage SET actual_units=:units, settled_at=now(),
                    reconciliation_evidence=:evidence, billable_units=CAST(:billable AS jsonb),
                    cost_source=:source, dispatch_state='settled'
                WHERE operation_id=:id AND principal=:principal
            '''),
                {
                    'id': operation_id,
                    'principal': principal,
                    'units': actual_units,
                    'evidence': evidence,
                    'billable': json.dumps(billable_units) if billable_units is not None else None,
                    'source': cost_source or ('administrative' if evidence else 'provider'),
                },
            )
        if actual_units > row['reserved_units']:
            # Record the debt before surfacing an invalid provider/tariff contract.
            raise BudgetError('Provider charge exceeded reservation; billing contract must be reviewed')

    async def rates(self):
        async with self.session_factory() as session:
            return (
                (
                    await session.execute(
                        text('''
                SELECT provider, model, version, unit, input_usd_per_million,
                       output_usd_per_million, context_tokens FROM cost_rates
                ORDER BY provider, model, version DESC
            ''')
                    )
                )
                .mappings()
                .all()
            )

    async def add_rate(self, rate):
        async with self.session_factory() as session, session.begin():
            result = await session.execute(
                text('''
                INSERT INTO cost_rates (provider, model, version, unit, input_usd_per_million,
                                        output_usd_per_million, context_tokens)
                VALUES (:provider, :model, :version, :unit, :input_usd_per_million,
                        :output_usd_per_million, :context_tokens)
                ON CONFLICT DO NOTHING RETURNING version
            '''),
                vars(rate),
            )
            if result.scalar_one_or_none() is None:
                raise BudgetError('Tariff version already exists; publish a new version')

    async def pending(self):
        return await self.usage(pending_only=True)

    async def usage(self, *, pending_only=False):
        async with self.session_factory() as session:
            return (
                (
                    await session.execute(
                        text('''
                SELECT operation_id, principal, provider, model, operation, rate_version,
                       reserved_units, provider_response_id, created_at, request_fingerprint,
                       dispatch_state, billable_units, cost_source, actual_units, settled_at,
                       reconciliation_evidence
                FROM cost_usage WHERE NOT :pending_only OR settled_at IS NULL
                ORDER BY CASE WHEN :pending_only THEN created_at END ASC, created_at DESC LIMIT 100
            '''),
                        {'pending_only': pending_only},
                    )
                )
                .mappings()
                .all()
            )

    async def mark_dispatch(self, principal, operation_id):
        async with self.session_factory() as session, session.begin():
            result = await session.execute(
                text("""UPDATE cost_usage SET dispatch_state='sending'
                    WHERE operation_id=:id AND principal=:principal
                    AND dispatch_state='reserved' RETURNING operation_id"""),
                {'id': operation_id, 'principal': principal},
            )
            if result.scalar_one_or_none() is None:
                raise BudgetError('Reservation is not available for dispatch')

    async def remember_response(self, principal, operation_id, response_id):
        async with self.session_factory() as session, session.begin():
            result = await session.execute(
                text('''UPDATE cost_usage SET provider_response_id=:response
                WHERE operation_id=:id AND principal=:principal
                AND (provider_response_id IS NULL OR provider_response_id=:response)
                RETURNING operation_id'''),
                {'response': response_id, 'id': operation_id, 'principal': principal},
            )
            if result.scalar_one_or_none() is None:
                raise BudgetError('Unknown reservation or conflicting provider response ID')

    async def reconcile_pending(self, operation_id, actual_units, evidence):
        async with self.session_factory() as session:
            principal = (
                await session.execute(
                    text('SELECT principal FROM cost_usage WHERE operation_id=:id'), {'id': operation_id}
                )
            ).scalar_one_or_none()
        if principal is None:
            raise BudgetError('Unknown reservation')
        await self.reconcile(principal, operation_id, actual_units, evidence=evidence)
