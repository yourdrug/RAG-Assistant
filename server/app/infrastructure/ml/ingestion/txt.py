"""Plain text parser."""

from __future__ import annotations

from pathlib import Path


def parse_txt(file_path: Path) -> tuple[str, dict]:
    """Parse TXT and return (text, metadata)."""
    return file_path.read_text(encoding="utf-8", errors="replace"), {}
