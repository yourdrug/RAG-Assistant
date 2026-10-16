"""Manual paid acceptance. Run with PYTHONPATH=app from server/.

Dispatch requires every model, a principal, an account label and a shared USD cap.
Verification reads exported provider confirmations and never performs inference.
A confirmation JSON object maps each operation ID to {usd, reference}.
"""

import argparse
import asyncio
from collections import Counter
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from application.budget import billing_principal
from config import settings
from domain.value_objects.cost_budget import BudgetError, usd_units
from domain.value_objects.llm_provider import LLMProvider
from infrastructure.benchmark.grouped_judge import create_async_judge_client, judge_batch
from infrastructure.benchmark.judge import close_judge_clients, get_judge_client, judge_with_structured_output
from infrastructure.budget.job_context import job_operations
from infrastructure.budget.ledger import PostgresRedisBudget
from infrastructure.budget.provider_catalog import OpenRouterBillingCatalog
from infrastructure.budget.runtime import budget_runtime
from infrastructure.ml.clients.deepinfra_clients import DeepInfraEmbeddingsClient, DeepInfraRerankerClient
from infrastructure.ml.clients.factories import create_openrouter_llm
from infrastructure.ml.clients.instructor_client import (
    create_async_instructor_client,
    create_instructor_client,
)
from infrastructure.ml.clients.managed_llm import ManagedLLM

log = logging.getLogger('accounting_acceptance')


class StructuredResult(BaseModel):
    value: int


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', required=True, help='Provider account label; never an API key')
    parser.add_argument('--principal', required=True, help='Authenticated user:<id> charged for this run')
    parser.add_argument('--since', help='UTC ISO timestamp of a previous dispatch; verification only')
    parser.add_argument('--confirmations', type=Path, help='Provider-exported operation -> {usd, reference}')
    parser.add_argument('--confirm-paid-calls', action='store_true')
    parser.add_argument('--maximum-usd')
    parser.add_argument('--llm-model', required=True)
    parser.add_argument('--judge-model', required=True)
    parser.add_argument('--embedding-model', required=True)
    parser.add_argument('--reranker-model', required=True)
    args = parser.parse_args()
    if not args.principal.startswith('user:') or len(args.principal) <= 5:
        parser.error('--principal must identify the authenticated user')
    if args.confirmations:
        if not args.since:
            parser.error('verification requires --since')
    elif (
        not args.confirm_paid_calls
        or not args.maximum_usd
        or not args.llm_model
        or not args.judge_model
        or not args.embedding_model
        or not args.reranker_model
    ):
        parser.error('dispatch requires --confirm-paid-calls, --maximum-usd and all four model flags')
    elif (
        usd_units(args.maximum_usd) <= 0
        or args.embedding_model != 'BAAI/bge-m3'
        or args.reranker_model != 'Qwen/Qwen3-Reranker-4B'
    ):
        parser.error('requires positive cap, BAAI/bge-m3 and Qwen/Qwen3-Reranker-4B')
    return args


def verify_confirmations(rows, confirmations):
    if not rows:
        raise BudgetError('No usage found for this principal and dispatch timestamp')
    for row in rows:
        confirmation = confirmations.get(row['operation_id'])
        if not confirmation or not str(confirmation.get('reference', '')).strip():
            raise BudgetError(f'Missing provider evidence for {row["operation_id"]}')
        if row['actual_units'] is None:
            raise BudgetError(f'Pending usage requires administrative reconciliation: {row["operation_id"]}')
        # Both values round up to integer nanodollars; permit one nanodollar of rounding.
        if abs(row['actual_units'] - usd_units(str(confirmation['usd']))) > 1:
            raise BudgetError(f'Provider confirmation differs for {row["operation_id"]}')
    return len(rows)


def verify_coverage(rows, args):
    expected = Counter()
    expected[('deepinfra', args.embedding_model, 'embedding')] += 3
    expected[('deepinfra', args.reranker_model, 'reranking')] += 2
    expected[('openrouter', args.llm_model, 'llm')] += 3
    expected[('openrouter', args.judge_model, 'llm')] += 2
    actual = Counter((row['provider'], row['model'], row['operation']) for row in rows)
    if any(actual[key] < count for key, count in expected.items()):
        raise BudgetError('Acceptance is missing required physical paths/models')


async def dispatch(args):
    settings.llm_provider = LLMProvider.OPENROUTER
    settings.cost_rate_limit_enabled = True
    settings.cost_hourly_limit = args.maximum_usd
    settings.cost_daily_limit = args.maximum_usd
    for provider, model in (
        ('deepinfra', args.embedding_model),
        ('deepinfra', args.reranker_model),
        ('openrouter', args.llm_model),
        ('openrouter', args.judge_model),
    ):
        await budget_runtime.rate(provider, model)
    embedding = DeepInfraEmbeddingsClient(
        settings.deepinfra_api_key,
        settings.deepinfra_base_url.rstrip("/") + "/openai",
        args.embedding_model,
    )
    reranker = DeepInfraRerankerClient(
        settings.deepinfra_api_key,
        settings.deepinfra_base_url,
        args.reranker_model,
    )
    llm = ManagedLLM(
        create_openrouter_llm(
            max_tokens=128,
            model=settings.model_copy(
                update={'openrouter_model': args.llm_model},
            ),
        )
    )
    structured = create_async_instructor_client(settings.openrouter_base_url, settings.openrouter_api_key)
    sync_structured = create_instructor_client(settings.openrouter_base_url, settings.openrouter_api_key)
    grouped = create_async_judge_client()
    try:
        # A bounded run completes within the rolling hour, so its shared quota is also the total run cap.
        async with asyncio.timeout(600):
            await embedding.embed_query('Accounting acceptance query')
            await embedding.embed_documents(['First accounting document', 'Second accounting document'])
            await asyncio.to_thread(embedding.embed_query_sync, 'Synchronous accounting query')
            await reranker.predict([('capital of France', 'Paris is the capital of France.')])
            await asyncio.to_thread(reranker.predict_sync, [('capital of France', 'Paris is in France.')])
            kwargs = {
                'model': args.llm_model,
                'max_tokens': 128,
                'max_retries': 1,
                'response_model': StructuredResult,
                'messages': [{'role': 'user', 'content': 'Return JSON with value equal to 1.'}],
            }
            await structured.chat.completions.create(**kwargs)
            await asyncio.to_thread(sync_structured.chat.completions.create, **kwargs)
            await llm.ainvoke('Reply with the single word OK.')
            judge = await asyncio.to_thread(get_judge_client, args.judge_model)
            await asyncio.to_thread(
                judge_with_structured_output,
                judge,
                'Score 10 if Paris is the capital of France; give a short reason.',
                args.judge_model,
            )
            saved = {}

            async def checkpoint(value):
                saved.update(value)

            await judge_batch(
                grouped,
                args.judge_model,
                {'question': 'Capital of France?', 'answer': 'Paris'},
                ['relevancy'],
                saved,
                checkpoint,
                budget=512,
                attempt=1,
            )
            if 'relevancy' not in saved:
                raise BudgetError('Grouped judge acceptance did not produce a score')
    finally:
        await asyncio.gather(
            embedding.close(),
            reranker.close(),
            llm.close(),
            structured.client.close(),
            grouped.close(),
            close_judge_clients(),
        )
        await asyncio.to_thread(sync_structured.client.close)


async def main(args):
    engine = create_async_engine(
        URL.create(
            'postgresql+asyncpg',
            username=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
            port=int(settings.db_port),
            database=settings.db_name,
        )
    )
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    ledger = PostgresRedisBudget(sessions, redis)
    budget_runtime.configure(ledger, sessions, OpenRouterBillingCatalog())
    token = billing_principal.set(args.principal)
    scope = job_operations.set((f'live:{args.account}:{uuid4().hex}', {}))
    since = datetime.fromisoformat(args.since) if args.since else datetime.now(UTC)
    if since.tzinfo is None:
        raise BudgetError('--since must include a UTC offset')
    try:
        if not args.confirmations:
            log.warning(
                'Paid dispatch account=%s since=%s cap_usd=%s',
                args.account,
                since.isoformat(),
                args.maximum_usd,
            )
            await dispatch(args)
        async with sessions() as session:
            rows = (
                (
                    await session.execute(
                        text('''
                SELECT operation_id, provider, model, operation, reserved_units, actual_units,
                       provider_response_id, billable_units, cost_source
                FROM cost_usage WHERE principal=:principal AND created_at >= :since ORDER BY created_at
            '''),
                        {'principal': args.principal, 'since': since},
                    )
                )
                .mappings()
                .all()
            )
        if args.confirmations:
            verify_coverage(rows, args)
            count = verify_confirmations(rows, json.loads(args.confirmations.read_text()))
            log.warning('Verified %d provider confirmations for account=%s', count, args.account)
        else:
            for row in rows:
                log.warning('Usage %s', json.dumps(dict(row)))
            log.warning(
                'Paid acceptance remains incomplete until every row has provider confirmation; since=%s',
                since.isoformat(),
            )
    finally:
        job_operations.reset(scope)
        billing_principal.reset(token)
        await redis.aclose()
        await engine.dispose()


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main(arguments()))
