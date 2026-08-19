"""Document domain classification — legal vs general."""

from __future__ import annotations

from enum import StrEnum


class DocDomain(StrEnum):
    GENERAL = "general"
    LEGAL = "legal"
