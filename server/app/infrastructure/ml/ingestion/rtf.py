"""RTF parsing — striprtf wrapper."""

from __future__ import annotations

from pathlib import Path

from striprtf.striprtf import rtf_to_text


def parse_rtf(file_path: Path) -> tuple[str, dict]:
    """Parse RTF and return (text, metadata)."""
    return rtf_to_text(file_path.read_text(encoding="utf-8", errors="replace")), {}
