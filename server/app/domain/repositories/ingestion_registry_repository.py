"""Ingestion Registry Repository interface."""

from __future__ import annotations

from typing import Protocol


class IngestionRegistryRepository(Protocol):
    def load(self) -> dict: ...
    def save(self, registry: dict) -> None: ...
    def is_already_indexed(self, key: str, file_hash: str) -> bool: ...
