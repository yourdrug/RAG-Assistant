"""Document domain classification — legal vs general vs decree."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal


class DocDomain(StrEnum):
    GENERAL = "general"
    LEGAL = "legal"
    DECREE = "decree"


class DomainDetectionMode(StrEnum):
    AUTO = "auto"


DocDomainSelection = Literal[DocDomain.GENERAL, DocDomain.LEGAL, DocDomain.DECREE, DomainDetectionMode.AUTO]
