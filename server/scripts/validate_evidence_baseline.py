"""Generate a checkpointed full RAG run, then judge the saved answers separately.

Input is an exported benchmark run. Historical results and database baselines are
never overwritten. Run from server with PYTHONPATH=app inside the service container.
"""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import httpx

from composition.container import Container
from config import _settings_overrides, settings
from domain.events.config_events import ConfigParameterChanged
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.roles import UserKind, UserRole
from infrastructure.benchmark.answer_generators import BenchmarkAnswer, RagBenchmarkGenerator
from infrastructure.benchmark.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.token_usage import judge_usage
from infrastructure.database.database import DatabaseManager
from infrastructure.events.in_process_event_bus import event_bus
from infrastructure.ml.rag.benchmark_evidence import BenchmarkEvidence
from infrastructure.ml.rag.evidence_focus import EVIDENCE_POLICY_VERSION
from infrastructure.ml.usage_capture import active_llm_usage, summarize_usage
from infrastructure.redis.redis_client import redis_client
from infrastructure.resilience.circuit_breaker import init_breakers

log = logging.getLogger('default')


class SavedGenerator:
    def __init__(self, generated):
        self.generated = generated

    async def generate(self, question, ctx):
        return self.generated


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


async def resolve_costs(records):
    pending = [r for r in records if r.get('cost_usd') is None and r.get('response_id')]
    if not pending:
        return
    async with httpx.AsyncClient(
        timeout=30, headers={'Authorization': 'Bearer ' + settings.openrouter_api_key}
    ) as client:
        for record in pending:
            try:
                response = await client.get(
                    settings.openrouter_base_url.rstrip('/') + '/generation',
                    params={'id': record['response_id']},
                )
                response.raise_for_status()
                data = response.json()['data']
                record['cost_usd'] = data.get('total_cost')
                record['provider'] = data.get('provider_name')
            except httpx.HTTPError:
                log.warning('Generation cost unavailable for %s', record['response_id'])


async def run(args):
    input_text = args.input.read_text()
    input_fingerprint = sha256(input_text.encode()).hexdigest()
    snapshot = json.loads(input_text)
    source = next(r for r in snapshot if r['id'] == args.source_run)
    cases = [
        {key: c.get(key) for key in ('id', 'question', 'source_hint', 'expected_answer', 'annotations')}
        for c in source['per_question_results']
    ]
    database = DatabaseManager()
    container = Container()
    await database.connect()
    await redis_client.init()
    try:
        container.init(database)
        async with container.infrastructure.db.uow.create() as uow:
            for row in await uow.config_parameters.get_all():
                event_bus.publish(
                    ConfigParameterChanged(
                        key=row.key,
                        old_value=None,
                        new_value=row.normalize(row.value),
                        value_type=row.value_type,
                        domain_key=row.domain_key,
                    )
                )
        # Pin the historical run's model without changing the persisted configuration.
        settings.openrouter_model = args.generation_model
        report = (
            json.loads(args.out.read_text())
            if args.out.exists()
            else {
                'source_run': args.source_run,
                'input_fingerprint': input_fingerprint,
                'evidence_policy': EVIDENCE_POLICY_VERSION,
                'model': settings.openrouter_model,
                'judge_model': args.judge_model,
                'config': {
                    'top_k': 8,
                    'fetch_k': 30,
                    'sparse_weight': 0.5,
                    'cache_enabled': False,
                    'max_concurrent': args.concurrency,
                },
                'cost_scope': 'Observed external model API costs; excludes local compute/hosting',
                'generated': {},
                'results': [],
            }
        )
        if (
            report['evidence_policy'] != EVIDENCE_POLICY_VERSION
            or report['model'] != settings.openrouter_model
            or report['source_run'] != args.source_run
            or report.get('input_fingerprint') != input_fingerprint
            or report['judge_model'] != args.judge_model
            or report['config']
            != {
                'top_k': 8,
                'fetch_k': 30,
                'sparse_weight': 0.5,
                'cache_enabled': False,
                'max_concurrent': args.concurrency,
            }
        ):
            raise ValueError(
                'Resume requires identical input, models, evidence policy and retrieval settings'
            )
        init_breakers(
            fail_max=settings.llm_breaker_fail_max, timeout_duration=settings.llm_breaker_timeout_duration
        )
        generator = RagBenchmarkGenerator(container.application.rag_service, 8, 30)
        ctx = ChatContext(1, UserKind.INTERNAL, UserRole.ADMIN)
        semaphore = asyncio.Semaphore(args.concurrency)

        async def generate(case):
            key = str(case['id'])
            if key in report['generated']:
                return
            async with semaphore:
                records = []
                usage_token = active_llm_usage.set(records)
                started = time.monotonic()
                overrides = _settings_overrides.set({'sparse_weight': 0.5})
                try:
                    answer = await generator.generate(case, ctx)
                except Exception as exc:
                    await resolve_costs(records)
                    report.setdefault('failures', []).append(
                        {
                            'question_id': case['id'],
                            'phase': 'generation',
                            'error_type': type(exc).__name__,
                            'llm_usage': records,
                            'rag_latency_sec': time.monotonic() - started,
                        }
                    )
                    save_report(args.out, report)
                    raise
                finally:
                    _settings_overrides.reset(overrides)
                    active_llm_usage.reset(usage_token)
                await resolve_costs(answer.llm_usage or [])
                report['generated'][key] = asdict(answer)
                save_report(args.out, report)
                log.warning(
                    'RAG %s done: %.2fs, %d model calls',
                    key,
                    answer.rag_latency_sec,
                    len(answer.llm_usage or []),
                )

        await complete_phase(generate(case) for case in cases)
        if args.generate_only:
            return

        async def evaluate(case):
            if any(r['id'] == case['id'] for r in report['results']):
                return
            async with semaphore:
                saved = dict(report['generated'][str(case['id'])])
                saved['evidence'] = BenchmarkEvidence(**saved['evidence']) if saved.get('evidence') else None
                evaluator = BenchmarkCaseEvaluator(SavedGenerator(BenchmarkAnswer(**saved)), args.judge_model)
                started = time.monotonic()
                records = []
                usage_token = judge_usage.set(records)
                try:
                    result = await evaluator.evaluate(0, case, 1, ctx)
                except Exception as exc:
                    await resolve_costs(records)
                    report.setdefault('failures', []).append(
                        {
                            'question_id': case['id'],
                            'phase': 'judge',
                            'error_type': type(exc).__name__,
                            'llm_usage': records,
                            'judge_latency_sec': time.monotonic() - started,
                        }
                    )
                    save_report(args.out, report)
                    raise
                finally:
                    judge_usage.reset(usage_token)
                result['judge_latency_sec'] = time.monotonic() - started
                result['latency_sec'] = result['rag_latency_sec'] + result['judge_latency_sec']
                await resolve_costs(result['judge_usage'])
                result['total_usage'] = summarize_usage(result['rag_usage'] + result['judge_usage'])
                result['total_cost_usd'] = result['total_usage']['cost_usd']
                result['cost_usd'] = result['total_cost_usd']
                report['results'].append(result)
                save_report(args.out, report)
                log.warning('Judge %s done: %.2fs', case['id'], result['judge_latency_sec'])

        await complete_phase(evaluate(case) for case in cases)
    finally:
        await container.dispose()
        await database.disconnect()
        await redis_client.aclose()


async def complete_phase(tasks):
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [result for result in results if isinstance(result, Exception)]
    if errors:
        raise ExceptionGroup('Benchmark phase failed; completed siblings were checkpointed', errors)
    cancellations = [result for result in results if isinstance(result, asyncio.CancelledError)]
    if cancellations:
        raise cancellations[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-run', type=int, default=12)
    parser.add_argument('--judge-model', default='qwen/qwen-2.5-7b-instruct')
    parser.add_argument('--generation-model', default='qwen/qwen-2.5-7b-instruct')
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--generate-only', action='store_true')
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error('concurrency must be positive')
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
