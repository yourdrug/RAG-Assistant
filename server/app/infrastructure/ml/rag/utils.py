"""Shared utilities for RAG helpers."""

from __future__ import annotations

from pathlib import Path


def clean_source_name(source: str) -> str:
    """Extract clean filename from full path, strip directory."""
    return Path(source).name if source else "unknown"
