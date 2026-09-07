"""Document source type — file upload vs manual chunk creation."""

from __future__ import annotations

from enum import StrEnum


class SourceType(StrEnum):
    FILE = "file"
    MANUAL = "manual"
