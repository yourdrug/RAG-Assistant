"""Plain text parser."""

from __future__ import annotations

from pathlib import Path

# Tried in order. utf-8-sig first so a BOM doesn't leak into the text as a
# literal character. cp1251/cp1252 cover the common legacy encodings for
# Russian/Western text; latin-1 never raises so it's always a safe last resort.
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "cp1252", "latin-1")


def _read_text_robust(file_path: Path) -> str:
    """Decode a text file trying encodings that round-trip cleanly.

    Forcing UTF-8 with errors="replace" silently corrupts every non-ASCII
    character in a non-UTF-8 file (mojibake), which then poisons search and
    embeddings. We instead try each candidate encoding strictly and keep the
    first one that decodes without error, falling back to a lossy decode
    only if nothing else works.
    """
    raw = file_path.read_bytes()
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return raw.decode("utf-8", errors="replace")


def parse_txt(file_path: Path) -> tuple[str, dict]:
    """Parse TXT and return (text, metadata)."""
    return _read_text_robust(file_path), {}
