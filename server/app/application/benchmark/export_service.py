"""Bounded report export from committed database results."""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import TYPE_CHECKING, Any

from domain.benchmark.evaluation import EVIDENCE_METRICS


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from application.ports.file_storage import FileStorage
    from application.uow import UnitOfWorkFactory

logger = logging.getLogger("default")
EXPORT_CHUNK_BYTES = 64 * 1024
CSV_FIELDS = (
    "id",
    "question",
    "answer",
    "faithfulness",
    "relevancy",
    "correctness",
    "hit_rate",
    "mrr",
    "context_precision",
    "context_recall",
    "latency_sec",
    "source_hit_rate",
    "source_mrr",
    "avg_sim",
    *EVIDENCE_METRICS,
    "context_chars",
)


def encode_json(value: object) -> Iterator[bytes]:
    for token in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(value):
        for offset in range(0, len(token), EXPORT_CHUNK_BYTES // 4):
            yield token[offset : offset + EXPORT_CHUNK_BYTES // 4].encode("utf-8")


def encode_csv(row: dict[str, Any] | None) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")

    if row is None:
        writer.writeheader()
    else:
        values = {
            **row.get("generator_metrics", {}),
            **row.get("retriever_metrics", {}),
            **row.get("context_metrics", {}),
            **row.get("evidence_metrics", {}),
            "context_chars": row.get("evidence", {}).get("context_chars"),
            "avg_sim": row.get("retriever_metrics", {}).get("avg_similarity", row.get("avg_similarity")),
            **row,
        }
        # Spreadsheet software must treat untrusted questions/answers as text.
        writer.writerow(
            {
                key: "'" + value
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
                else value
                for key, value in values.items()
                if key in CSV_FIELDS
            }
        )

    return stream.getvalue().encode("utf-8")


def encode_result(row: dict[str, Any], report_format: str, *, first: bool) -> Iterator[bytes]:
    if report_format == "json":
        yield b"," if not first else b""
        yield from encode_json(row)
    else:
        encoded = encode_csv(row)

        for start in range(0, len(encoded), EXPORT_CHUNK_BYTES):
            yield encoded[start : start + EXPORT_CHUNK_BYTES]


def validate_report_format(report_format: str) -> None:
    if report_format not in ("csv", "json"):
        message = "Export format must be csv or json"
        raise ValueError(message)


class BenchmarkExportService:
    def __init__(self, uow_factory: UnitOfWorkFactory, storage: FileStorage | None = None) -> None:
        self.uow_factory = uow_factory
        self.storage = storage

    async def header(self, run_id: int) -> dict[str, Any]:
        async with self.uow_factory.create(master=True) as uow:
            header = await uow.benchmark_runs.export_header(run_id)

        if header is None:
            message = "Benchmark result not found"
            raise ValueError(message)

        return header

    async def export(self, run_id: int, report_format: str) -> AsyncIterator[bytes]:
        validate_report_format(report_format)

        header = await self.header(run_id)

        if report_format == "json":
            yield b'{"run":'

            for block in encode_json(header):
                yield block

            yield b',"results":['
        else:
            yield encode_csv(None)

        offset = 0
        first = True

        while True:
            # Release the read transaction before yielding to HTTP/S3 consumers.
            async with self.uow_factory.create(master=True) as uow:
                rows = await uow.benchmark_runs.result_page(run_id, offset=offset, limit=1)

            if not rows:
                break

            row = rows[0]

            for block in encode_result(row, report_format, first=first):
                yield block

            first = False
            offset += 1

        if report_format == "json":
            yield b"]}"

    async def save_report(self, run_id: int, report_format: str) -> bool:
        """Artifact failure is independent of evaluation; explicit retries reuse its key."""
        validate_report_format(report_format)

        if self.storage is None:
            message = "Report storage is required"
            raise RuntimeError(message)

        await self.header(run_id)
        key = f"benchmarks/runs/{run_id}/report.{report_format}"
        metadata: dict[str, str | int | None] = {"key": key, "status": "pending", "size_bytes": None}
        async with self.uow_factory.create(master=True) as uow:
            await uow.benchmark_runs.set_artifact(run_id, report_format, metadata)
        size = 0

        async def blocks() -> AsyncIterator[bytes]:
            nonlocal size

            async for block in self.export(run_id, report_format):
                size += len(block)
                yield block

        try:
            await self.storage.upload_stream(key, blocks())
        except Exception:
            logger.exception("Benchmark report upload failed: run=%s format=%s", run_id, report_format)
            metadata["status"] = "failed"
        else:
            metadata.update(status="ready", size_bytes=size)
        async with self.uow_factory.create(master=True) as uow:
            await uow.benchmark_runs.set_artifact(run_id, report_format, metadata)
        return metadata["status"] == "ready"
