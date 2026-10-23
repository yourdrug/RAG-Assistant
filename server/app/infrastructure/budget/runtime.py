"""Provider admission shared by chat, auxiliary calls and benchmark adapters."""

from sqlalchemy import text
from infrastructure.database.database import database
from domain.runtime.config import parse_bool

from application.budget import BudgetPolicy, CostBudgetService
from config import settings
from domain.runtime.cost import BudgetError, CostRate
import asyncio


class BudgetRuntime:
    def __init__(self):
        self.service = None
        self.session_factory = None
        self.catalog = None
        self.bootstrapped = False
        self.loop = None
        self.database = database

    def configure(self, ledger, session_factory, catalog, *, database_manager=database) -> None:
        self.loop = asyncio.get_running_loop()
        self.service = CostBudgetService(ledger)
        self.session_factory = session_factory
        self.catalog = catalog
        self.database = database_manager

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

    def policy_sync(self) -> BudgetPolicy:
        if not self.bootstrapped:
            return BudgetPolicy(
                settings.cost_rate_limit_enabled,
                str(settings.cost_hourly_limit),
                str(settings.cost_daily_limit),
            )

        with self.database.master_connection() as connection:
            connection.execute(text("SET statement_timeout = '5s'"))
            rows = connection.execute(text('''SELECT key, value FROM config_parameters
                WHERE domain_key IS NULL AND key IN
                ('cost_rate_limit_enabled','cost_hourly_limit','cost_daily_limit')''')).all()
        return self.decode_policy(rows)

    async def rate(self, provider, model):
        if self.session_factory is None:
            raise BudgetError('Cost backend is not initialized')

        async with self.session_factory() as session:
            row = (
                (
                    await session.execute(
                        text('''
                SELECT provider, model, version, unit, input_usd_per_million,
                       output_usd_per_million, context_tokens,
                       provider = 'deepinfra' AS input_only FROM cost_rates
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
