"""Chunk search mode."""

from __future__ import annotations

from enum import StrEnum


class SearchMode(StrEnum):
    EXACT = "exact"
    ICONTAINS = "icontains"
