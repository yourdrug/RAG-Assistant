"""Explicit, non-destructive import of legacy JSON reports and JSONL summaries."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from domain.benchmark.run import BenchmarkRun
from infrastructure.benchmark.metrics import compute_summary_metrics

MAX_LEGACY_BYTES = 64 * 1024 * 1024


def load_legacy_runs(path: Path, dataset: str) -> tuple[list[BenchmarkRun], list[str]]:
    if path.stat().st_size > MAX_LEGACY_BYTES:
        raise ValueError('Legacy input exceeds 64 MiB; split the input before importing')
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines() if path.suffix == '.jsonl' else [text]
    runs = []
    errors = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, parse_constant=reject_constant)
            digest = hashlib.sha256(
                json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            metrics, config, questions, timestamp = parse_record(record, path)
            validate_legacy_metrics(metrics)
            # Include dataset in identity: the same content can belong to different datasets.
            identity = hashlib.sha256(f'{dataset}:{digest}'.encode()).hexdigest()
            runs.append(
                BenchmarkRun(
                    config_json={
                        **config,
                        'legacy_import': {'source': path.name, 'sha256': digest, 'record': number},
                    },
                    summary_metrics=metrics,
                    per_question_results=questions,
                    duration_sec=float(metrics.get('total_time_sec') or 0),
                    llm_evaluated=any(
                        metrics.get(key) is not None
                        for key in ('avg_faithfulness', 'faithfulness', 'avg_answer_correctness')
                    ),
                    dataset=dataset,
                    legacy_hash=identity,
                    creation_date=timestamp,
                )
            )
        except (ValueError, TypeError, KeyError, OverflowError):
            # Do not echo report content or private answers in diagnostics.
            errors.append(f'Record {number}: invalid JSON/report schema')
    return runs, errors


def reject_constant(value: str):
    raise ValueError('Non-finite JSON number')


def parse_record(record: object, path: Path) -> tuple[dict, dict, list[dict] | None, datetime]:
    questions: list[dict] | None
    if isinstance(record, list):
        if not record or not all(isinstance(row, dict) for row in record):
            raise ValueError('Report must contain question objects')
        metrics = compute_summary_metrics(record)
        config = {}
        questions = record
        timestamp = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    elif isinstance(record, dict):
        metrics = record.get('metrics', record.get('summary_metrics'))
        config = record.get('config', record.get('config_json', {}))
        questions = record.get('per_question_results')
        if not isinstance(metrics, dict) or not isinstance(config, dict):
            raise ValueError('Expected metrics and config objects')
        if questions is not None and (
            not isinstance(questions, list) or not all(isinstance(row, dict) for row in questions)
        ):
            raise ValueError('Invalid per-question results')
        if not record.get('timestamp'):
            raise ValueError('Summary requires its original timestamp')
        timestamp = datetime.fromisoformat(record['timestamp'])
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
    else:
        raise ValueError('Expected report array or summary object')
    return metrics, config, questions, timestamp


def validate_legacy_metrics(metrics: dict) -> None:
    for key, value in metrics.items():
        numeric = key.startswith(("avg_", "total_")) or key in {
            "hit_rate",
            "faithfulness",
            "relevancy",
            "correctness",
        }
        if numeric and value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError('Invalid numeric metric')
