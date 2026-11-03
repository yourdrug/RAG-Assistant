"""Provider admission shared by chat, auxiliary calls and benchmark adapters."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from application.budget import BudgetPolicy, CostBudgetService
from application.runtime.config import parse_bool
from config import settings
from domain.runtime.cost import BudgetError, CostRate
from infrastructure.database.database import database


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Iterable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from application.budget import BillingCatalogPort, CostBudgetPort
    from infrastructure.database.database import DatabaseManager


class BudgetRuntime:
    def __init__(self) -> None:
        self.service: CostBudgetService | None = None
        self.session_factory: Callable[[], AsyncSession] | None = None
        self.catalog: BillingCatalogPort | None = None
        self.bootstrapped = False
        self.loop: asyncio.AbstractEventLoop | None = None
        self.database = database

    def configure(
        self,
        ledger: CostBudgetPort,
        session_factory: Callable[[], AsyncSession],
        catalog: BillingCatalogPort,
        *,
        database_manager: DatabaseManager = database,
    ) -> None:
        self.loop = asyncio.get_running_loop()
        self.service = CostBudgetService(ledger)
        self.session_factory = session_factory
        self.catalog = catalog
        self.database = database_manager

    def run_sync[**P, T](
        self, function: Callable[P, Coroutine[Any, Any, T]], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Use the Composition-owned async resources from a worker thread."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            msg = "Synchronous paid clients must run outside the event loop"
            raise BudgetError(msg)

        if self.loop is None or not self.loop.is_running():
            msg = "Cost backend event loop is unavailable"
            raise BudgetError(msg)

        return asyncio.run_coroutine_threadsafe(function(*args, **kwargs), self.loop).result()

    def decode_policy(self, rows: Iterable[Sequence[str]]) -> BudgetPolicy:
        values = {row[0]: row[1] for row in rows}

        if not {"cost_rate_limit_enabled", "cost_hourly_limit", "cost_daily_limit"} <= values.keys():
            msg = "Budget configuration is not initialized"
            raise BudgetError(msg)

        return BudgetPolicy(
            parse_bool(values["cost_rate_limit_enabled"]),
            values["cost_hourly_limit"],
            values["cost_daily_limit"],
        )

    async def policy_snapshot(self) -> BudgetPolicy:
        if not self.bootstrapped:
            return BudgetPolicy(
                settings.cost_rate_limit_enabled,
                str(settings.cost_hourly_limit),
                str(settings.cost_daily_limit),
            )

        if self.session_factory is None:
            msg = "Cost backend is not initialized"
            raise BudgetError(msg)

        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    text("""SELECT key, value FROM config_parameters
                WHERE domain_key IS NULL AND key IN
                ('cost_rate_limit_enabled','cost_hourly_limit','cost_daily_limit')""")
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
            rows = connection.execute(
                text("""SELECT key, value FROM config_parameters
                WHERE domain_key IS NULL AND key IN
                ('cost_rate_limit_enabled','cost_hourly_limit','cost_daily_limit')""")
            ).all()
        return self.decode_policy(rows)

    async def rate(self, provider: str, model: str) -> CostRate:
        if self.session_factory is None:
            msg = "Cost backend is not initialized"
            raise BudgetError(msg)

        async with self.session_factory() as session:
            row = (
                (
                    await session.execute(
                        text("""
                SELECT provider, model, version, unit, input_usd_per_million,
                       output_usd_per_million, context_tokens,
                       provider = 'deepinfra' AS input_only FROM cost_rates
                WHERE provider=:provider AND model=:model ORDER BY version DESC LIMIT 1
            """),
                        {"provider": provider, "model": model},
                    )
                )
                .mappings()
                .one_or_none()
            )

        if row is None:
            msg = "Unknown provider/model price; paid call blocked"
            raise BudgetError(msg)

        rate = CostRate(**row)

        if self.catalog is None or rate.context_tokens < await self.catalog.max_context(provider, model):
            msg = "Provider context bound changed; publish a verified tariff version"
            raise BudgetError(msg)

        async with self.session_factory() as session:
            violated = (
                await session.execute(
                    text("""SELECT EXISTS(SELECT 1 FROM cost_usage
                WHERE provider=:provider AND model=:model AND rate_version=:version
                AND actual_units > reserved_units)"""),
                    {"provider": provider, "model": model, "version": rate.version},
                )
            ).scalar_one()

        if violated:
            msg = "This tariff exceeded its reservation bound; paid call blocked"
            raise BudgetError(msg)

        return rate
