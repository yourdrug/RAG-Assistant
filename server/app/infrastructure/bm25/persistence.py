"""BM25 index persistence — filesystem and S3 save/load."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from infrastructure.bm25.bm25_index import BM25Index

log = logging.getLogger("default")

BM25_S3_KEY = "_system/bm25_index.json"


def save_bm25_index(index: BM25Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(index.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
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
    data = json.dumps(index.to_dict(), ensure_ascii=False).encode()
    await storage.upload_file(BM25_S3_KEY, data)
    log.info(
        "BM25 index saved to S3: %d docs -> s3://%s", index.n_docs, BM25_S3_KEY
    )


async def load_bm25_index_from_s3(storage) -> BM25Index | None:
    """Load BM25 index from S3. Returns None if not found or parse fails."""
    try:
        temp_path = await storage.download_to_temp(BM25_S3_KEY)
        try:
            data = json.loads(temp_path.read_text(encoding="utf-8"))
            idx = BM25Index.from_dict(data)
            log.info("BM25 index loaded from S3: %d docs", idx.n_docs)
            return idx
        finally:
            temp_path.unlink(missing_ok=True)
    except Exception as e:
        log.warning("Failed to load BM25 index from S3: %s", e)
        return None


def load_bm25_index_from_s3_sync(storage) -> BM25Index | None:
    """Load BM25 index from S3 synchronously."""
    try:
        raw = storage.download_bytes(BM25_S3_KEY)
        data = json.loads(raw)
        idx = BM25Index.from_dict(data)
        log.info("BM25 index loaded from S3 (sync): %d docs", idx.n_docs)
        return idx
    except Exception as e:
        log.warning("Failed to load BM25 index from S3 (sync): %s", e)
        return None
