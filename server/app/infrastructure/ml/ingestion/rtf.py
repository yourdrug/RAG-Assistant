"""RTF parsing — striprtf wrapper."""

from __future__ import annotations

from pathlib import Path

from striprtf.striprtf import rtf_to_text


def _read_rtf_text(file_path: Path) -> str:
    """Read raw RTF bytes, trying encodings that preserve the RTF structure.

    RTF files from older Russian systems are often saved as Windows-1251.
    Reading them as UTF-8 with errors="replace" corrupts RTF escape sequences
    and produces garbled output.  We try utf-8 first (most common for modern
    files), then fall back to cp1251, and finally latin-1 (never fails).
    """
    raw = file_path.read_bytes()
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return raw.decode("latin-1")


def parse_rtf(file_path: Path) -> tuple[str, dict]:
    """Parse RTF and return (text, metadata)."""
    return rtf_to_text(_read_rtf_text(file_path)), {}
