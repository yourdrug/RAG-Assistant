"""Document domain classification — legal vs general vs decree."""

from __future__ import annotations

from enum import StrEnum


class DocDomain(StrEnum):
    GENERAL = "general"
    LEGAL = "legal"
    DECREE = "decree"
