"""BM25 index persistence — filesystem and S3 save/load."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
from aiohttp import ClientError as HTTPClientError

from application.ports.file_storage import FileStorage
from config import settings

from infrastructure.bm25.bm25_index import BM25Index
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.storage.file_storage import S3_MISSING_CODES, run_storage_io

log = logging.getLogger("default")


class BM25LoadError(RuntimeError):
    """Snapshot load failed; absence must never mask storage/corruption errors."""


class BM25SnapshotNotReady(RuntimeError):
    """A CPU consumer must prepare the snapshot asynchronously before use."""


BM25_S3_KEY = "_system/bm25_index.json"


def save_bm25_index(index: BM25Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index.to_dict(), ensure_ascii=False), encoding="utf-8")
    log.info("BM25 index saved: %d docs -> %s", index.n_docs, path)


def load_bm25_index(path: Path) -> BM25Index | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        idx = BM25Index.from_dict(data)
        log.info("BM25 index loaded: %d docs from %s", idx.n_docs, path)
        return idx
    except Exception as e:
        log.warning("Failed to load BM25 index from %s: %s", path, e)
        return None


async def save_bm25_index_to_s3(index: BM25Index, storage) -> None:
    """Save BM25 index to S3 under the system prefix."""
    data = await asyncio.to_thread(lambda: json.dumps(index.to_dict(), ensure_ascii=False).encode())
    await storage.upload_file(BM25_S3_KEY, data)
    log.info("BM25 index saved to S3: %d docs -> s3://%s", index.n_docs, BM25_S3_KEY)


def parse_bm25_snapshot(path: Path) -> BM25Index:
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_snapshot(data)
    return BM25Index.from_dict(data)


async def load_bm25_index_from_s3(storage: FileStorage) -> BM25Index | None:
    """Load and parse off loop; only confirmed absence yields no snapshot."""
    try:
        temp_path = await storage.download_to_temp(BM25_S3_KEY)
        cancelled = False
        try:
            index = await run_storage_io(parse_bm25_snapshot, temp_path)
            log.info("BM25 index loaded from S3: %d docs", index.n_docs)
            return index
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            try:
                await run_storage_io(temp_path.unlink, missing_ok=True)
            except OSError:
                if not cancelled:
                    raise
                # Cleanup failure must not replace request cancellation.
                log.warning("Failed to remove BM25 snapshot after cancellation: %s", temp_path, exc_info=True)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in S3_MISSING_CODES:
            return None
        raise BM25LoadError("Failed to load BM25 snapshot from S3") from exc
    except (BotoCoreError, HTTPClientError, OSError, ValueError, KeyError, TypeError) as exc:
        raise BM25LoadError("Failed to read/parse BM25 snapshot") from exc


def validate_snapshot_column(field: str, values, count: int, value_type: type) -> None:
    """Validate parallel snapshot columns before they reach scoring/ACL code."""
    if values is None:
        return  # Legacy snapshots omit hashes/ACL and retain existing defaults.
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"BM25 snapshot {field} must match texts length")
    for value in values:
        if value is None and field != "hashes":
            continue
        if not isinstance(value, value_type) or isinstance(value, bool):
            raise ValueError(f"Invalid BM25 snapshot {field} value")
        if field == "doc_visibility" and isinstance(value, str):
            DocumentVisibility(value)


def validate_snapshot(data) -> None:
    """Reject malformed persisted data without catching unrelated programming errors."""
    if not isinstance(data, dict) or not isinstance(data.get("texts"), list):
        raise ValueError("BM25 snapshot texts must be a list")
    texts = data["texts"]
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("BM25 snapshot texts must contain strings")
    for field, value_type in (
        ("hashes", str),
        ("doc_visibility", str),
        ("doc_owner_id", int),
        ("doc_group_id", int),
    ):
        validate_snapshot_column(field, data.get(field), len(texts), value_type)
    for field in ("k1", "b"):
        if field not in data:
            continue
        value = data[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"BM25 snapshot {field} must be a finite number")
        if value < 0 or (field == "b" and value > 1):
            raise ValueError(f"Invalid BM25 snapshot {field} range")


async def load_configured_bm25_snapshot(storage: FileStorage) -> BM25Index | None:
    """Retain the S3-only sparse capability of the existing runtime registry."""
    if settings.file_backend != "s3":
        return None
    return await load_bm25_index_from_s3(storage)
