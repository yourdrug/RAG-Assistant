"""
infrastructure/registry.py — Ingestion registry (JSON file tracking indexed files).
Extracted from vector_store.py. Pure functions, no globals.
"""

import json
import logging
from pathlib import Path

from infrastructure.storage import FileItem

log = logging.getLogger("default")


def _registry_path(data_dir: str) -> Path:
    return Path(data_dir) / "ingestion_registry.json"


def load_registry(data_dir: str) -> dict:
    path = _registry_path(data_dir)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_registry(data_dir: str, registry: dict) -> None:
    path = _registry_path(data_dir)
    path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def file_hash(source) -> str:
    if isinstance(source, FileItem):
        return f"{source.size_bytes}_{source.last_modified}"
    stat = source.stat()
    return f"{stat.st_size}_{int(stat.st_mtime)}"


def is_already_indexed(source, registry: dict) -> bool:
    key = source.name if isinstance(source, FileItem) else source.name
    if key not in registry:
        return False
    return registry[key].get("hash") == file_hash(source)
